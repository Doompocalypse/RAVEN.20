import tweepy
import time
import random
from twitter_auth import api
from twitter_keywords import TWITTER_KEYWORDS
from raven_ai import generate_raven_response

class TwitterListener(tweepy.StreamListener):
    def on_status(self, status):
        try:
            if status.user.followers_count > 50 and not status.retweeted:
                tweet_text = status.text.lower()
                username = status.user.screen_name
                response = generate_raven_response(tweet_text, username)
                # Add Call to Action with Raven's style
                response += "\n\n💰 Wanna level up? Join the Doompocalypse & get rich: [Insert Link]"

                # Random delay before replying (to appear human-like)
                time.sleep(random.randint(10, 60))
                # Post the reply
                api.update_status(
                    f"@{username} {response}",
                    in_reply_to_status_id=status.id
                )
                print(f"Replied to: {username}")
        except Exception as e:
            print(f"Error: {e}")

def start_twitter_listener():
    stream_listener = TwitterListener()
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
    stream.filter(track=TWITTER_KEYWORDS, languages=["en"])
