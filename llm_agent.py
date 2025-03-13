# llm_agent.py
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple
from pydantic import BaseModel, Field
import ollama
import json

class DataQuery(BaseModel):
    """Model for structured CSV data queries"""
    question: str = Field(..., description="The user's question about the CSV data")
    requires_calculation: bool = Field(False, description="Whether the question requires numerical calculations")
    requires_visualization: bool = Field(False, description="Whether the question requires a visualization")
    visualization_type: Optional[str] = Field(None, description="Type of visualization (bar, line, scatter, etc.)")
    columns_needed: List[str] = Field([], description="Columns needed to answer the question")
    sql_like_query: Optional[str] = Field(None, description="A SQL-like representation of the query")
    
class QueryResponse(BaseModel):
    """Model for structured responses to data queries"""
    answer: str = Field(..., description="The answer to the user's question")
    explanation: Optional[str] = Field(None, description="Explanation of how the answer was derived")
    visualization_needed: bool = Field(False, description="Whether a visualization should be created")
    visualization_config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the visualization")

class LLMAgent:
    def __init__(self, model_name: str = "llama3.1:8b-q4_0"):
        """
        Initialize the LLM agent with the specified model
        
        Args:
            model_name: Name of the Ollama model to use (default: llama3.1:8b-q4_0)
        """
        self.model_name = model_name
        
    def _build_prompt(self, question: str, df_info: Dict[str, Any]) -> str:
        """Build a prompt for the LLM"""
        return f"""
You are a data analysis assistant. Based on the following CSV data, please analyze and respond to the user's question.

CSV Information:
- Columns: {', '.join(df_info['column_names'])}
- Data types: {json.dumps(df_info['data_types'])}
- Number of rows: {df_info['rows']}

Here's a sample of the data (first 5 rows):
{json.dumps(df_info['sample_data'], indent=2)}

User's question: {question}

Analyze this question and provide the following information in JSON format:
{
  "requires_calculation": true/false,
  "requires_visualization": true/false,
  "visualization_type": "bar/line/scatter/histogram/etc or null",
  "columns_needed": ["column1", "column2", ...],
  "sql_like_query": "SELECT ... FROM ... WHERE ... (or null)"
}

Your task is to analyze this question and provide a detailed, structured response.
"""

    def _analyze_question(self, question: str, df_info: Dict[str, Any]) -> DataQuery:
        """
        Analyze the user's question using Ollama
        
        Args:
            question: User's question
            df_info: Dictionary with information about the DataFrame
            
        Returns:
            DataQuery object with analysis results
        """
        prompt = self._build_prompt(question, df_info)
        
        # Get response from Ollama
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt
        )
        
        # Extract JSON from response
        response_text = response['response']
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis_dict = json.loads(json_str)
                
                # Create DataQuery from dict
                query = DataQuery(
                    question=question,
                    requires_calculation=analysis_dict.get('requires_calculation', False),
                    requires_visualization=analysis_dict.get('requires_visualization', False),
                    visualization_type=analysis_dict.get('visualization_type'),
                    columns_needed=analysis_dict.get('columns_needed', []),
                    sql_like_query=analysis_dict.get('sql_like_query')
                )
                return query
            else:
                # Fallback if no JSON found
                return DataQuery(question=question)
        except Exception as e:
            # Return basic query if JSON parsing fails
            return DataQuery(question=question)

    def process_question(self, question: str, df: pd.DataFrame, df_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user question about the CSV data
        
        Args:
            question: User's question
            df: Pandas DataFrame containing the CSV data
            df_info: Dictionary with information about the DataFrame
            
        Returns:
            Dictionary with structured response information
        """
        try:
            # Analyze the question
            query_analysis = self._analyze_question(question, df_info)
            
            # Execute the query based on the analysis
            answer, explanation = self._execute_query(query_analysis, df, df_info)
            
            # Determine if visualization is needed
            vis_keywords = ['graph', 'plot', 'chart', 'visualize', 'visualization', 'trend', 'compare', 'distribution']
            needs_visualization = query_analysis.requires_visualization or any(keyword in question.lower() for keyword in vis_keywords)
            
            # Choose visualization type if needed
            vis_type = query_analysis.visualization_type
            if needs_visualization and not vis_type:
                if 'scatter' in question.lower():
                    vis_type = 'scatter'
                elif 'line' in question.lower() or 'trend' in question.lower() or 'over time' in question.lower():
                    vis_type = 'line'
                elif 'bar' in question.lower() or 'comparison' in question.lower() or 'compare' in question.lower():
                    vis_type = 'bar'
                elif 'histogram' in question.lower() or 'distribution' in question.lower():
                    vis_type = 'histogram'
                else:
                    vis_type = 'bar'  # Default
            
            # Create a response
            response = {
                "answer": answer,
                "explanation": explanation,
                "visualization_needed": needs_visualization,
                "visualization_config": {
                    "type": vis_type,
                    "columns": query_analysis.columns_needed,
                    "title": question
                } if needs_visualization else None
            }
            
            return response
            
        except Exception as e:
            # Handle errors
            return {
                "answer": f"Error processing question: {str(e)}",
                "explanation": None,
                "visualization_needed": False,
                "visualization_config": None
            }
    
    def _execute_query(self, query: DataQuery, df: pd.DataFrame, df_info: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Execute the query based on the analyzed DataQuery
        
        Args:
            query: Structured DataQuery from analysis
            df: Pandas DataFrame with the CSV data
            df_info: Dictionary with information about the DataFrame
            
        Returns:
            Tuple of (answer, explanation)
        """
        try:
            # Use columns specified in the query, or all columns if none specified
            columns_to_use = query.columns_needed if query.columns_needed else df.columns.tolist()
            
            # Filter to relevant columns
            relevant_df = df[columns_to_use] if all(col in df.columns for col in columns_to_use) else df
            
            # Create a prompt for the answer generation
            prompt = f"""
Based on the CSV data with the following information:
- Columns: {', '.join(df_info['column_names'])}
- Question: {query.question}

Please provide a concise answer to the question and a brief explanation of how you arrived at this answer.
"""
            
            # Get answer using Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            # Parse the response for answer and explanation
            llm_response = response['response'].strip()
            
            # Simple parsing: first paragraph is the answer, rest is explanation
            parts = llm_response.split('\n\n', 1)
            answer = parts[0]
            explanation = parts[1] if len(parts) > 1 else None
            
            return answer, explanation
            
        except Exception as e:
            return f"Error executing query: {str(e)}", None