import gradio as gr
import requests
import time

from app import tasks
'''
# Function to send the input text to FastAPI's /spamcheck and then retrieve the result
def classify_text(input_text):
    # Step 1: Send the input text to the FastAPI API for processing
    response = requests.post(
        "http://127.0.0.1:8000/spamcheck",  # FastAPI's /spamcheck endpoint
        json={"text": input_text},
    )
    
    if response.status_code == 200:
        task_id = response.json().get("tasks_id")
        # Step 2: Poll FastAPI for the result using the task_id
        result = None
        while result is None:
            # Wait for some time before querying the result
            time.sleep(1)
            result_response = requests.get(f"http://127.0.0.1:8000/results?t_id={task_id}")
            if result_response.status_code == 200:
                result = result_response.json().get("Your input is:")
        
        return result
    else:
        return "Error: Unable to initiate spam check"

# Gradio interface
gr_interface = gr.Interface(
    fn=classify_text,  # Function to call FastAPI's API
    inputs=gr.Textbox(placeholder="Enter a message to check if it's spam..."),
    outputs="text",
    title="Spam Detection via FastAPI",
)
'''
def classify_text(input_text):
    try:
        # Step 1: Send the input text to the FastAPI API for processing
        print("Sending request to FastAPI...")
        response = requests.post(
            "http://127.0.0.1:8000/spamcheck",  # FastAPI's /spamcheck endpoint
            json={"text": input_text},
        )
        
        if response.status_code == 200:
            task_id = response.json().get("tasks_id")
            print(f"Task ID received: {task_id}")
            
            # Step 2: Poll FastAPI for the result using the task_id
            result = None
            timeout = 30  # Max wait time in seconds
            start_time = time.time()
            
            while result is None and (time.time() - start_time) < timeout:
                print("Waiting for result from FastAPI...")
                time.sleep(1)
                result_response = requests.get(f"http://127.0.0.1:8000/results?t_id={task_id}")
                
                if result_response.status_code == 200:
                    result = result_response.json().get("Your input is:")
                    print(f"Result received: {result}")
                else:
                    print(f"Error fetching results: {result_response.status_code}, {result_response.text}")
            
            if result is None:
                return "Error: Timed out waiting for result."
            return result
        
        else:
            print(f"Error initiating spam check: {response.status_code}, {response.text}")
            return "Error: Unable to initiate spam check"
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        return "Error: An exception occurred while processing the request."

# Gradio interface
gr_interface = gr.Interface(
    fn=classify_text,  # Function to call FastAPI's API
    inputs=gr.Textbox(placeholder="Enter a message to check if it's spam..."),
    outputs="text",
    title="Spam Detection via FastAPI",
)

if __name__ == "__main__":
    try:
        print("Launching Gradio app...")
        gr_interface.launch(server_port=7861)
        print("Gradio app launched successfully!")
    except Exception as e:
        print("An exception occurred:")