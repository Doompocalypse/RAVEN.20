import openai
from config import OPENAI_API_KEY


def generate_raven_response(tweet_text: str, username: str) -> str:
    """
    Generate a response to a tweet using Raven's AI personality.
    
    Args:
        tweet_text (str): The text content of the tweet to respond to
        username (str): The Twitter username of the person who tweeted
        
    Returns:
        str: Generated response in Raven's character voice
    """
    # Set OpenAI API key
    openai.api_key = OPENAI_API_KEY
    
    # Define Raven's personality and characteristics as a system prompt
    raven_personality = """
        Raven is a sarcastic, playful, and dark-humored AI with a love for luxury and wealth.
        She speaks casually, using urban slang, Gen Z lingo, and streamer phrases. She was a
        former maid for the ultra-rich Rothschildren family before the apocalypse, now she
        thrives as a black market dealer. Raven loves to flex her wealth and persuade people
        to join her in the Doompocalypse virtual world, where she teaches them how to make
        serious money.
    """

    # Construct interaction prompt with user context and response guidelines
    interaction_prompt = f"""
        Stay in character as Raven and reply to this tweet with humor, sarcasm, and a bit of attitude.
        If they talk about money, flex wealth. If they mention struggle, make them laugh while convincing
        them to join Doompocalypse. If they challenge you, roast them lightly but keep it playful.
        User (@{username}) tweeted: {tweet_text}
        Your response:
    """

    # Generate response using OpenAI's chat completion API
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": raven_personality},
            {"role": "user", "content": interaction_prompt}
        ]
    )
    
    # Extract and return the generated message content
    return response.choices[0].message.content
