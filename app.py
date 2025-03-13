# app.py
import gradio as gr
import pandas as pd
from csv_handler import CSVHandler
from llm_agent import LLMAgent
from visualizer import Visualizer

# Initialize components
csv_handler = CSVHandler()
llm_agent = LLMAgent(model_name="llama3.1:8b-q4_0")  # Adjust model as needed
visualizer = Visualizer()

# Global variables
csv_data = None
csv_info = None

def upload_csv(file_obj):
    """Handle CSV file upload"""
    global csv_data, csv_info
    
    success, error_msg, file_info = csv_handler.load_csv(file_obj)
    if success:
        csv_data = csv_handler.get_dataframe()
        csv_info = file_info
        
        # Create a summary of the data
        summary = f"✅ Successfully loaded CSV file\n\n"
        summary += f"📊 {file_info['rows']} rows × {file_info['columns']} columns\n\n"
        summary += "📋 Column preview:\n"
        
        for col in file_info['column_names']:
            dtype = file_info['data_types'][col]
            summary += f"- {col} ({dtype})\n"
            
        return summary
    else:
        return f"❌ Error: {error_msg}"

def process_question(question):
    """Process a user question about the CSV data"""
    global csv_data, csv_info
    
    if csv_data is None:
        return "Please upload a CSV file first.", None
    
    # Process the question using the LLM agent
    response = llm_agent.process_question(question, csv_data, csv_info)
    
    # Create visualization if needed
    if response["visualization_needed"] and response["visualization_config"]:
        fig, error = visualizer.create_visualization(csv_data, response["visualization_config"])
        if fig:
            return response["answer"], fig
        else:
            return f"{response['answer']}\n\nCould not create visualization: {error}", None
    else:
        return response["answer"], None

# Create Gradio interface
with gr.Blocks(title="CSV Question Answering & Visualization") as demo:
    gr.Markdown("# CSV Question Answering and Visualization")
    gr.Markdown("Upload a CSV file, ask questions about the data, and get AI-powered answers with visualizations.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CSV File (max 25MB)")
            file_info = gr.Textbox(label="File Information", lines=10, interactive=False)
            
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="Ask a question about your data", placeholder="e.g., What is the average price? Show me a histogram of prices.")
            submit_btn = gr.Button("Ask", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(label="Answer", lines=8, interactive=False)
                
            with gr.Column():
                plot_output = gr.Plot(label="Visualization")
    
    # Set up event handlers
    file_input.upload(upload_csv, inputs=[file_input], outputs=[file_info])
    submit_btn.click(process_question, inputs=[question_input], outputs=[answer_output, plot_output])
    question_input.submit(process_question, inputs=[question_input], outputs=[answer_output, plot_output])
    
# Launch the app
if __name__ == "__main__":
    demo.launch()