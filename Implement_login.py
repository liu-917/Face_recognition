import tkinter as tk
from tkinter import messagebox
import sqlite3
from PIL import Image, ImageTk
import Face_recognition_entry
import Face_recognition_training
import Face_recognition_recognition
def init_db():
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def login():
    user = entry_username.get()
    pwd = entry_password.get()
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
    if c.fetchone():
        messagebox.showinfo('welcome',f'欢迎{user}')
        open_new_window()
    else:
        messagebox.showerror("The username or password is incorrect!")
    conn.close()

def register():
    user = entry_username.get()
    pwd = entry_password.get()
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user, pwd))
        conn.commit()
        messagebox.showinfo("The user registration is successful")
    except sqlite3.IntegrityError:
        messagebox.showerror("The user already exists")
    finally:
        conn.close()


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    window.geometry(f'{width}x{height}+{x}+{y}')



def open_new_window():
    new_window = tk.Tk()
    app.destroy()
    new_window.title("Face recognition")
    new_window.geometry("400x300")
    center_window(new_window,400,300)


    btn_face_registration = tk.Button(new_window, text="Face entry", command=run_face_registration)
    btn_face_registration.pack()

    btn_face_training = tk.Button(new_window, text="Model training", command=run_face_training)
    btn_face_training.pack()

    btn_face_recognition = tk.Button(new_window, text="Start identifying", command=run_face_recognition)
    btn_face_recognition.pack()

def run_face_recognition():

    Face_recognition_recognition.rlsb()
    pass


def run_face_registration():

    Face_recognition_entry.entry()
    pass


def run_face_training():

    Face_recognition_training.train()
    pass

app = tk.Tk()
center_window(app,400,300)
app.title("login")

processed_image = Image.open("picture.jpg")
processed_image=processed_image.resize((100,100))
processed_photo = ImageTk.PhotoImage(processed_image)

image_label = tk.Label(app, image=processed_photo)
image_label.place(x=0, y=0)

tk.Label(app, text="Username:").pack()
entry_username = tk.Entry(app)
entry_username.pack()

tk.Label(app, text="password:").pack()
entry_password = tk.Entry(app, show="*")
entry_password.pack()

tk.Button(app, text="login", command=login).pack()
tk.Button(app, text="enroll", command=register).pack()

app.mainloop()
