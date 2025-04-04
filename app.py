import tkinter as tk
from tkinter import ttk, messagebox
from reconocimiento.reconocimiento import reconocer_señas_en_tiempo_real

class AplicacionLenguajeSeñas:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación de Lenguaje de Señas")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')
        
        self.configurar_interfaz()
    
    def configurar_interfaz(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Encabezado
        lbl_titulo = tk.Label(
            main_frame,
            text="Aplicación de Lenguaje de Señas",
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        lbl_titulo.pack(pady=(0, 20))
        
        lbl_subtitulo = tk.Label(
            main_frame,
            text="Por un mundo más inclusivo",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        lbl_subtitulo.pack(pady=(0, 30))
        
        # Botones de opciones
        btn_estilo = {
            'font': ('Arial', 12),
            'width': 25,
            'height': 2,
            'bd': 0,
            'highlightthickness': 0,
            'activebackground': '#3498db',
            'activeforeground': 'white'
        }
        
        
        btn_reconocer = tk.Button(
            main_frame,
            text="Reconocer señas en tiempo real",
            bg='#2ecc71',
            fg='white',
            command=self.reconocer_senas,
            **btn_estilo
        )
        btn_reconocer.pack(pady=10)
        
        btn_salir = tk.Button(
            main_frame,
            text="Salir",
            bg='#e74c3c',
            fg='white',
            command=self.salir_aplicacion,
            **btn_estilo
        )
        btn_salir.pack(pady=10)
        
        # Footer
        lbl_footer = tk.Label(
            main_frame,
            text="© 2023 Aplicación de Lenguaje de Señas",
            font=('Arial', 8),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        lbl_footer.pack(side='bottom', pady=20)
    
    
    def reconocer_senas(self):
        self.root.withdraw()  # Ocultar ventana principal
        reconocer_señas_en_tiempo_real()
        self.root.deiconify()  # Mostrar ventana principal nuevamente
    
    def salir_aplicacion(self):
        if messagebox.askyesno(
            "Salir", 
            "¿Estás seguro que deseas salir de la aplicación?"
        ):
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionLenguajeSeñas(root)
    root.mainloop()