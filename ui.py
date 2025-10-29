import gradio as gr
import utils
import tools
import os

os.environ['CREWAI_DISABLE_TELEMETRY']='true'
os.environ['CREWAI_DISABLE_TRACING']='true'
os.environ['CREWAI_TELEMETRY']='false'
os.environ['OTEL_SDK_DISABLED']='true'

def show_section():
    # Show the hidden section and disable the button
    return gr.update(visible=True), gr.update(interactive=False)

def update_visit_history(r_type, r_price, r_rating, user_rating, review_title, review_text, images):
    user_visit_history = []
    with open('synthetic-visit-histories/synthetic_visit_histories.txt', 'r') as f:
        for line in f:
            user_visit_history.append(line.strip()) # Remove newline characters
    new_visit_history = utils.add_user_visit(r_type, r_price, r_rating, user_rating, 
                                            review_title, review_text,images)
    user_visit_history.append(new_visit_history)

    #Update the history txt
    if len(user_visit_history) > 10:
        user_visit_history.pop(0)
    with open('synthetic-visit-histories/synthetic_visit_histories.txt', 'w') as f:
        for item in user_visit_history:
            f.write(f"{item}\n")

    return (gr.update(value="Added Successfully! You may add a new one!"),
            gr.update(value=[]),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value="")
    )

css = """
.centered-markdown {
    text-align: center;
    display: block; /* Ensure the element behaves as a block for text-align to work */
}

#my_submit_button {
      background-color: #0000FF; /* Blue */
      color: white;
}

#visit_submit_button {
      background-color: #0000FF; /* Blue */
      color: white;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("## Your Personalized Restaurant Recommender", elem_classes="centered-markdown")
    show_button = gr.Button("If you would like to add a recent restaurant visit, CLICK here!", elem_id="my_submit_button")

    with gr.Row(equal_height=True, visible=False) as hidden_section:
        with gr.Column():
            gr.Markdown("## Upload photos of your recent restaurant visits (Optional)")
            images = gr.Gallery(type='filepath')

        with gr.Column():
            gr.Markdown("## Reviews and Info about the recent restaurant visit")

            with gr.Column():
                r_type = gr.Dropdown(["Restaurant", "Cafe", "Bar"], label="Restaurant Type")
                r_price = gr.Textbox(label='Restaurant Price Tier (0 (unknown), 1 ($), 2 ($$-$$$), 3 ($$$$))')
                r_rating = gr.Textbox(label='Average Restaurant Rating (0-5)')
                user_rating = gr.Textbox(label='Your rating on the restaurant (0-5)')
                review_title = gr.Textbox(label='The title of your review (short review)')
                review_text = gr.Textbox(label='The text of your review (long review)')
                add_history_button = gr.Button("Add a Visit History to the Existing Histories (will only keep the 10 most recent visits)",elem_id="visit_submit_button")
    
    show_button.click(show_section, outputs=[hidden_section, show_button])

    add_history_button.click(update_visit_history,
                 inputs=[r_type, r_price, r_rating, user_rating, review_title, review_text,images], 
                 outputs=[add_history_button,images,r_price,r_rating,user_rating,review_title,review_text])

    gr.Markdown("## It's time to ask for the suggestion!")

    crew = tools.crew_builup()

    def recommend(message, history):
        user_visit_history = []
        with open('synthetic-visit-histories/synthetic_visit_histories.txt', 'r') as f:
            for line in f:
                user_visit_history.append(line.strip()) # Remove newline characters
        
        inputs = {"visit_history": user_visit_history,
                "user_question": message}

        return str(crew.kickoff(inputs=inputs))
    
    gr.ChatInterface(
        fn=recommend, 
        type="messages"
    )

if __name__ == "__main__":
    demo.launch(share=True)
