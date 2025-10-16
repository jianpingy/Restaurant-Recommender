import gradio as gr
import utils

with gr.Blocks() as demo:
    gr.Markdown("## Restaurant Recommender")
    images = gr.Gallery(type='filepath')
    r_type = gr.Dropdown(["Restaurant", "Cafe", "Bar"], label="Restaurant Type")
    r_price = gr.Textbox(label='Restaurant Price Tier (0 (unknown), 1 ($), 2 ($$-$$$), 3 ($$$$))')
    r_rating = gr.Textbox(label='Average Restaurant Rating (0-5)')
    user_rating = gr.Textbox(label='Your rating on the restaurant (0-5)')
    review_title = gr.Textbox(label='The title of your review (short review)')
    review_text = gr.Textbox(label='The text of your review (long review)')
    submit = gr.Button("Add a visit history")
    new_user_visit = gr.Textbox(label="New User Visit History", lines=10, interactive=False)

    submit.click(utils.add_user_visit, 
                 inputs=[r_type, r_price, r_rating, user_rating, review_title, review_text,images], 
                 outputs=[new_user_visit])

if __name__ == "__main__":
    demo.launch(share=True)