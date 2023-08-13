import sys
sys.path.append('../..')

import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from decode_str import decode_str as decode

def translate_sentence():
    # Retrieve user inputs
    sentence = text_input.get("1.0", "end-1c")
    beam_size = beam_size_input.get()
    max_steps = max_steps_input.get()
    top_k = top_k_input.get()
    model_path = model_path_input.get("1.0", "end-1c")

    # Perform translation
    try:
        # Load the model and perform translation using the given parameters
        print(type(top_k))
        translation_result = decode(model_path,int(top_k),int(beam_size),int(max_steps),sentence)
        print(translation_result)
        # Display the translation result
        result_text.delete("1.0", "end")
        result_text.insert("1.0", translation_result)
    except Exception as e:
        messagebox.showerror("Translation Error", str(e))

def insert_default_path():
    default_model_path = "../../logs/hiddenLayer1_bs16/checkpoints2/model_8.pt"
    model_path_input.delete("1.0", "end")
    model_path_input.insert("1.0", default_model_path)

# Create the main window
window = tk.Tk()
window.title("Translation GUI")

# Create labels and input fields
text_label = tk.Label(window, text="Text:")
text_label.pack()
text_input = tk.Text(window, height=5, width=40)
text_input.pack()

beam_size_label = tk.Label(window, text="Beam Size:")
beam_size_label.pack()
beam_size_input = tk.Entry(window)
beam_size_input.pack()

max_steps_label = tk.Label(window, text="Max Decoding Time Steps:")
max_steps_label.pack()
max_steps_input = tk.Entry(window)
max_steps_input.pack()

top_k_label = tk.Label(window, text="Top K Results:")
top_k_label.pack()
top_k_input = tk.Entry(window)
top_k_input.pack()

model_path_label = tk.Label(window, text="Model Path:")
model_path_label.pack()
model_path_input = tk.Text(window, height=2, width=40)
model_path_input.pack()

default_path_button = tk.Button(window, text="Insert Default Path", command=insert_default_path)
default_path_button.pack()

translate_button = tk.Button(window, text="Translate", command=translate_sentence)
translate_button.pack()

result_label = tk.Label(window, text="Translation Result:")
result_label.pack()
result_text = tk.Text(window, height=5, width=40)
result_text.pack()

# Run the GUI
window.mainloop()
