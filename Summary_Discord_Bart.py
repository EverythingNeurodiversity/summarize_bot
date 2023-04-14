import discord
from discord.ext import commands
from transformers import BartTokenizer, BartForConditionalGeneration

# Create a bot instance
bot = commands.Bot(command_prefix='!')

# Load the BART model and tokenizer for text summarization
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Define a dictionary to store the timestamp of the last message read by the user in each channel
last_read = {}

# Define a function to summarize text using the BART model
def summarize_text(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@bot.event
async def on_message(message):
    global last_read
    # Update the timestamp of the last message read by the user in the current channel
    last_read[message.channel.id] = message.created_at
    await bot.process_commands(message)

@bot.command()
async def summarize(ctx):
    global last_read
    # Get the timestamp of the last message read in the current channel
    last_read_time = last_read.get(ctx.channel.id)
    if not last_read_time:
        await ctx.send("No previous messages to summarize.")
        return
    # Retrieve messages since the last read timestamp
    messages = await ctx.channel.history(after=last_read_time).flatten()
    message_text = '\n'.join([msg.content for msg in messages])
    # Summarize the messages using the BART model
    summary = summarize_text(message_text)
    await ctx.send(summary)

# Run the bot
bot.run('KEY')
