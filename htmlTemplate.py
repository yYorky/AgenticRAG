css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 100px;
  max-height: 100px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

# Templates for each chatbot with distinct avatars
bot_template_1 = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="http://localhost:8501/static/York_AI_1.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template_2 = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="http://localhost:8501/static/York_AI_2.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template_3 = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="http://localhost:8501/static/York_AI_3.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="http://localhost:8501/static/York.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
