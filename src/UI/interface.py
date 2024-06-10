import sys
import os
import customtkinter as ctk
import json
import tkinter as tk

def get_user_inputs():
    inputs = {}
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'user_inputs.json')
    icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')

    def submit():
        try:
            inputs["benchmarks"] = [key for key, var in selected_benchmarks.items() if var.get()]
            for param in params:
                if param in ["mutation_rate", "final_mutation_rate", "mutation_strength", "final_mutation_strength"]:
                    inputs[param] = float(entries[param].get())
                else:
                    inputs[param] = int(entries[param].get())
            with open(config_path, "w") as f:
                json.dump(inputs, f)
            app.quit()  # Fechar a janela da interface gráfica após a submissão
        except ValueError:
            error_label.configure(text="Please enter valid numbers for all fields")

    def on_closing():
        # Fecha o programa se a janela for fechada sem clicar em "Submit"
        app.destroy()
        os._exit(0)

    app = ctk.CTk()

    # Centraliza a janela na tela
    window_width, window_height = 400, 700
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    app.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    app.title("ProTεuS G.A.O.")
    app.iconbitmap(icon_path)  # Adiciona o ícone
    app.resizable(False, False)  # Torna a janela não redimensionável

    app.protocol("WM_DELETE_WINDOW", on_closing)

    ctk.CTkLabel(app, text="ProTεuS Genetic Algorithm Optimization", font=("Segoe UI", 20)).pack(pady=10)
    ctk.CTkLabel(app, text="Select the Benchmark Functions", font=("Segoe UI", 14)).pack(pady=5)

    benchmarks_frame = ctk.CTkFrame(app)
    benchmarks_frame.pack(pady=5, padx=20, anchor="center")

    benchmarks = ["RASTRIGIN", "ACKLEY", "SPHERE", "EASOM", "MCCORMICK"]
    selected_benchmarks = {}
    for benchmark in benchmarks:
        selected_benchmarks[benchmark] = ctk.BooleanVar()
        ctk.CTkCheckBox(benchmarks_frame, text=benchmark, variable=selected_benchmarks[benchmark], font=("Segoe UI", 12)).pack(anchor="w", pady=2)

    ctk.CTkLabel(app, text="Function Parameters", font=("Segoe UI", 14)).pack(pady=5)

    entries_frame = ctk.CTkFrame(app)
    entries_frame.pack(pady=5, padx=20)

    entries = {}
    params = {
        "population_size": "Population Size",
        "number_of_generations": "Number of Generations",
        "dimensions": "Dimensions",
        "mutation_rate": "Mutation Rate",
        "final_mutation_rate": "Final Mutation Rate",
        "mutation_strength": "Mutation Strength",
        "final_mutation_strength": "Final Mutation Strength",
        "nTests": "Number of Tests"
    }
    for param, label in params.items():
        frame = ctk.CTkFrame(entries_frame)
        frame.pack(pady=5)
        ctk.CTkLabel(frame, text=label, font=("Segoe UI", 12)).pack(side="left", padx=10)
        entries[param] = ctk.CTkEntry(frame)
        entries[param].pack(side="left", padx=10)

    ctk.CTkButton(app, text="Submit", command=submit, font=("Segoe UI", 12)).pack(pady=20)

    error_label = ctk.CTkLabel(app, text="", text_color="red", font=("Segoe UI", 12))
    error_label.pack(pady=5)

    # Adiciona o texto "Alpha_1.0" no canto inferior esquerdo
    version_label = ctk.CTkLabel(app, text="Alpha_1.3", font=("Segoe UI", 10))
    version_label.place(relx=0.01, rely=1.01, anchor='sw')

    app.mainloop()

if __name__ == "__main__":
    get_user_inputs()