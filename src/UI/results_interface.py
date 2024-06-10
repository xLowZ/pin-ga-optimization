import os
import json
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

def show_results():
    results_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'results.json')

    with open(results_path, "r") as f:
        all_results = json.load(f)

    app = ctk.CTk()
    app.title("GA Benchmark Results")
    app.geometry("700x900")
    app.resizable(False, False)

    # Centralizar a janela na tela
    window_width = 700
    window_height = 900
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    app.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    def save_results():
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as file:
                for benchmark in all_results:
                    benchmark_name = benchmark["benchmark_name"]
                    file.write(f"{benchmark_name}\n")
                    results = benchmark["results"]
                    for result in results:
                        test_label = f"TEST {result['test_number']}"
                        file.write(f"{test_label}\n")
                        file.write(f"Mean of solutions: {result['mean_value']}\n")
                        file.write(f"Standard deviation of solutions: {result['std_value']}\n")
                        file.write(f"Best Genes found: {result['best_genes']}\n")
                        file.write(f"Best Fitness found: {result['best_value']}\n")
                        file.write(f"Worst solution found: {result['worst_value']}\n")
                        file.write("-" * 40 + "\n")
                    file.write("=" * 40 + "\n\n")

    # Menu
    menu_bar = tk.Menu(app)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Save Results", command=save_results)
    menu_bar.add_cascade(label="File", menu=file_menu)
    app.config(menu=menu_bar)

    # Frame principal com barra de rolagem
    scrollable_frame = ctk.CTkScrollableFrame(app, width=680, height=880)
    scrollable_frame.pack(pady=10, padx=10, fill="both", expand=True)

    def display_results():
        for benchmark in all_results:
            benchmark_name = benchmark["benchmark_name"]
            results = benchmark["results"]
            
            # Frame para a caixa ao redor do nome do benchmark
            benchmark_frame = ctk.CTkFrame(scrollable_frame, corner_radius=10, fg_color="#333333")
            benchmark_frame.pack(pady=20, padx=20, fill="x")

            ctk.CTkLabel(benchmark_frame, text=benchmark_name, font=("Segoe UI", 24, "bold")).pack(pady=10, padx=10)
            
            for result in results:
                test_label = f"TEST {result['test_number']}"
                ctk.CTkLabel(scrollable_frame, text=test_label, font=("Segoe UI", 18, "bold")).pack(pady=10)
                ctk.CTkLabel(scrollable_frame, text=f"Mean of solutions: {result['mean_value']}", font=("Segoe UI", 14)).pack(pady=2)
                ctk.CTkLabel(scrollable_frame, text=f"Standard deviation of solutions: {result['std_value']}", font=("Segoe UI", 14)).pack(pady=2)
                ctk.CTkLabel(scrollable_frame, text=f"Best solution Genes found: {result['best_genes']}", font=("Segoe UI", 14)).pack(pady=2)
                ctk.CTkLabel(scrollable_frame, text=f"Best solution Fitness found: {result['best_value']}", font=("Segoe UI", 14)).pack(pady=2)
                ctk.CTkLabel(scrollable_frame, text=f"Worst solution found: {result['worst_value']}", font=("Segoe UI", 14)).pack(pady=2)
                
                # Adicionar um separador simples
                separator = ctk.CTkFrame(scrollable_frame, height=2, width=650, fg_color="grey")
                separator.pack(pady=15)

            # Adicionar um espaço maior entre diferentes benchmarks
            ctk.CTkFrame(scrollable_frame, height=20).pack()

    display_results()
    app.mainloop()

if __name__ == "__main__":
    show_results()
