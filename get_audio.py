from openai import OpenAI

client = OpenAI(api_key='sk-proj-TVRJgZLveipYXfH6ufQZT3BlbkFJslnLIAYVY6vh6LEeO22e')


def get_tiny_text(instructions):
    content = """
    You are an helpful assistant. I will give to you list of string. 
    Your task is to re-write the elements of the list in a single, fancy string.
    Return ONLY the fancy string.
    """
    instructions = ', '.join(instructions)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": instructions}
        ],
        temperature=0.9
    )
    result = response.choices[0].message.content
    return result


def get_audio(instructions):
    text = get_tiny_text(instructions)

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    response.stream_to_file("output.mp3")
    return response
