```markdown
# Poetic Muse: AI-Powered Poem Generator

Welcome to Poetic Muse, an AI-powered poem generator application that uses deep learning to create poetry. This application is built with TensorFlow and Tkinter, and it generates poems based on user input.

## Features

- Generate poems using an LSTM model trained on a collection of poems.
- Rhyming functionality to ensure that every 5th word in the poem rhymes.
- User-friendly GUI for easy interaction.

## Installation

### Prerequisites

- Python 3.x
- TensorFlow
- NLTK
- NumPy
- Pillow
- Tkinter

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/poetic-muse.git
   cd poetic-muse
   ```

2. **Install Required Libraries**

   ```bash
   pip install numpy tensorflow nltk pillow
   ```

3. **Download NLTK Data**

   ```python
   import nltk
   nltk.download('cmudict')
   ```

4. **Prepare Your Data**

   - **Poems:** Place your `.txt` files containing poems in a folder named `haiku` (or adjust the `folder_path` variable in the code to point to your folder).

5. **Load Pretrained Model**

   - Ensure that you have a trained LSTM model saved as `lstm_poem_model.keras` in the project directory. If you don't have a model, you need to train one or use a pre-trained model.

6. **Add Background Image**

   - Place your background image named `background.jpg` in the project directory or adjust the image path in the code.

## Usage

1. **Run the Application**

   ```bash
   python app.py
   ```

2. **Generate a Poem**

   - Enter a seed text in the input field and click the "Generate" button.
   - The generated poem will be displayed in the output area.

## Code Overview

- `read_poems_from_folder(folder_path)`: Reads and concatenates poems from text files in a specified folder.
- `get_rhyming_words(word, word_list)`: Finds rhyming words for a given word from a list.
- `generate_text(seed_text, next_words, model, max_sequence_len)`: Generates a poem using the LSTM model.
- `PoemGeneratorApp`: The Tkinter-based GUI application for interacting with the poem generator.

## Notes

- Ensure that the model and tokenizer are correctly set up before running the application.
- The `generate_text` function includes a rhyme feature that generates a rhyming word every 5th position in the poem.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [NLTK](https://www.nltk.org/)
- [Tkinter](https://wiki.python.org/moin/TkInter)
- [Pillow](https://python-pillow.org/)

For further questions or contributions, feel free to open an issue or submit a pull request.

Happy Poeting!
```