css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex; 
    align-items: center; /* Ensures proper vertical alignment */
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    flex-shrink: 0; /* Prevent the avatar from resizing */
    width: 200px; /* Matches the avatar size */
    margin-right: 1rem; /* Adds space between avatar and message */
}
.chat-message .avatar img {
    max-width: 200px;
    max-height: 200px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    flex-grow: 1; /* Ensures the message takes up the remaining space */
    padding: 0 1.5rem;
    color: #fff;
}
</style>

'''

# Templates for each chatbot with distinct avatars
bot_template_1 = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/yYorky/AgenticRAG/refs/heads/main/static/York_AI_1.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template_2 = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/yYorky/AgenticRAG/refs/heads/main/static/York_AI_2.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template_3 = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/yYorky/AgenticRAG/refs/heads/main/static/York_AI_3.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/yYorky/AgenticRAG/refs/heads/main/static/York.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
