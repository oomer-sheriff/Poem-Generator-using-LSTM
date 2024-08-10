import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import cmudict
import os
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
def read_poems_from_folder(folder_path):
    all_poems = ""
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Read the contents of each file
            with open(file_path, 'r', encoding='utf-8') as file:
                poem_content = file.read()
                
            # Add a newline between poems and append to the main string
            all_poems += poem_content + "\n\n"
    
    return all_poems
# [Include all the functions from your original script here: read_poems_from_folder, get_rhyming_words, generate_text]

def get_rhyming_words(word, word_list):
    if word.lower() not in pronouncing_dict:
        return []
    
    pronunciations = pronouncing_dict[word.lower()]
    rhymes = []
    
    for w in word_list:
        if w.lower() in pronouncing_dict:
            w_pronunciations = pronouncing_dict[w.lower()]
            for pron1 in pronunciations:
                for pron2 in w_pronunciations:
                    if pron1[-2:] == pron2[-2:]:
                        rhymes.append(w)
                        break
                if w in rhymes:
                    break
    
    return rhymes
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        if (_ + 1) % 5 == 0:  # Every 5th word should rhyme
            last_word = seed_text.split()[-5]
            rhyming_words = get_rhyming_words(last_word, list(tokenizer.word_index.keys()))
            if rhyming_words:
                rhyming_probs = [predicted_probs[tokenizer.word_index[w]] for w in rhyming_words if w in tokenizer.word_index]
                if rhyming_probs:
                    predicted_word = rhyming_words[np.argmax(rhyming_probs)]
                else:
                    predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
            else:
                predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        else:
            predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        
        seed_text += " " + predicted_word
    
    return seed_text
# Load the model and prepare the tokenizer
folder_path = "haiku"
text = read_poems_from_folder(folder_path)

nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

max_sequence_len = max([len(tokenizer.texts_to_sequences([line])[0]) for line in text.split('\n')])

model = tf.keras.models.load_model('lstm_poem_model.keras')

class PoemGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Poetic Muse")
        master.geometry("600x800")
        master.configure(bg="#1E1E1E")

        self.title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
        self.text_font = tkfont.Font(family="Georgia", size=12)

        self.setup_ui()

    def setup_ui(self):
        # Load and display background image
        bg_image = Image.open("background.jpg")  # Replace with your image path
        bg_image = bg_image.resize((600, 800))
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(self.master, image=self.bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Title
        title_label = tk.Label(self.master, text="Poetic Muse", font=self.title_font, bg="#1E1E1E", fg="#FFD700")
        title_label.pack(pady=20)

        # Input field
        self.input_frame = tk.Frame(self.master, bg="#2C2C2C", bd=5)
        self.input_frame.pack(pady=10)

        self.input_entry = tk.Entry(self.input_frame, font=self.text_font, width=30, bg="#3E3E3E", fg="white", insertbackground="white")
        self.input_entry.pack(side=tk.LEFT, padx=5)

        self.generate_button = tk.Button(self.input_frame, text="Generate", command=self.generate_poem, font=self.text_font, bg="#4CAF50", fg="white")
        self.generate_button.pack(side=tk.LEFT, padx=5)

        # Output area
        self.output_frame = tk.Frame(self.master, bg="#2C2C2C", bd=5)
        self.output_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(self.output_frame, font=self.text_font, wrap=tk.WORD, bg="#3E3E3E", fg="white")
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def generate_poem(self):
        title = self.input_entry.get()
        if not title:
            return

        generated_text = generate_text(title, 90, model, max_sequence_len)
        formatted_poem = self.format_poem(title, generated_text)

        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, formatted_poem)

    def format_poem(self, title, text):
        lines = text.split()
        formatted_lines = [title.upper()]
        formatted_lines.append("")  # Blank line after title
        
        current_line = ""
        for i, word in enumerate(lines):
            if i > 0 and i % 5 == 0:
                formatted_lines.append(current_line.strip())
                current_line = ""
            current_line += word + " "
        
        if current_line:
            formatted_lines.append(current_line.strip())

        return "\n".join(formatted_lines)

if __name__ == "__main__":
    root = tk.Tk()
    app = PoemGeneratorApp(root)
    root.mainloop()