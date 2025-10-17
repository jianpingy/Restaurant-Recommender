import gradio as gr
import utils
import tools

with gr.Blocks() as demo:
    gr.Markdown("## Restaurant Recommender")
    images = gr.Gallery(type='filepath')
    print(images)
    r_type = gr.Dropdown(["Restaurant", "Cafe", "Bar"], label="Restaurant Type")
    r_price = gr.Textbox(label='Restaurant Price Tier (0 (unknown), 1 ($), 2 ($$-$$$), 3 ($$$$))')
    r_rating = gr.Textbox(label='Average Restaurant Rating (0-5)')
    user_rating = gr.Textbox(label='Your rating on the restaurant (0-5)')
    review_title = gr.Textbox(label='The title of your review (short review)')
    review_text = gr.Textbox(label='The text of your review (long review)')
    submit = gr.Button("Add a visit history")
    new_user_visit = gr.Textbox(label="Final Recommendation", lines=10, interactive=False)
    
    # user_visit_history = []
    # with open('synthetic-visit-histories/synthetic_visit_histories.txt', 'r') as f:
    #     for line in f:
    #         user_visit_history.append(line.strip()) # Remove newline characters

    # new_visit_history = utils.add_user_visit(r_type, r_price, r_rating, user_rating, 
    #                                          review_title, review_text,images)
    # user_visit_history.append(new_visit_history)

    crew = tools.crew_builup()
    
    def recommend(r_type, r_price, r_rating, user_rating, review_title, review_text, images):
        user_visit_history = []
        with open('synthetic-visit-histories/synthetic_visit_histories.txt', 'r') as f:
            for line in f:
                user_visit_history.append(line.strip()) # Remove newline characters
        new_visit_history = utils.add_user_visit(r_type, r_price, r_rating, user_rating, 
                                             review_title, review_text,images)
        user_visit_history.append(new_visit_history)
        
        inputs = {"visit_history": user_visit_history}

        return crew.kickoff(inputs=inputs) 

    # submit.click(recommend,
    #              inputs=[crew,inputs],
    #              outputs=[final_recommendation])

    # submit.click(utils.add_user_visit, 
    #              inputs=[r_type, r_price, r_rating, user_rating, review_title, review_text,images], 
    #              outputs=[new_user_visit])

    submit.click(recommend, 
                 inputs=[r_type, r_price, r_rating, user_rating, review_title, review_text, images], 
                 outputs=[new_user_visit])

if __name__ == "__main__":
    demo.launch(share=True)