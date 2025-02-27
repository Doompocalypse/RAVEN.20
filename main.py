import threading
from twitter_listener import start_twitter_listener

def start_chatbot():
    print("Starting Raven AI chatbot with Twitter engagement...")
    # Start Twitter Listener in a separate thread
    twitter_thread = threading.Thread(target=start_twitter_listener)
    twitter_thread.daemon = True
    twitter_thread.start()
    # Continue chatbot's existing functionality
    # chatbot_main_loop()

if __name__ == "__main__":
    start_chatbot()