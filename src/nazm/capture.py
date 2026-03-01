import tkinter as tk
from pathlib import Path
import os
import mss
import numpy as np
import cv2
import uuid

class RegionSelector:
    def __init__(self):
        self.root = tk.Tk()

        with mss.mss() as sct:
            all_monitors = sct.monitors[0]
            self.width = all_monitors["width"]
            self.height = all_monitors["height"]
            self.left = all_monitors["left"]
            self.top = all_monitors["top"]

        # 2. Configurar a janela para o tamanho total
        # O formato é: LARGURAxALTURA+X+Y
        self.root.geometry(f"{self.width}x{self.height}+{self.left}+{self.top}")
        self.root.overrideredirect(True) # Remove barra de título e bordas
        self.root.attributes("-topmost", True)

        # DEFINIÇÃO DA TRANSPARÊNCIA MÁGICA
        # Escolhemos uma cor que não usaremos em outro lugar
        TRANS_COLOR = '#abcdef' 
        self.root.attributes('-transparentcolor', TRANS_COLOR)

        # O Canvas agora é preenchido com a cor que ficará semi-transparente
        # Usamos uma cor escura para o efeito de overlay
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Deixamos a janela inteira com a opacidade do overlay
        self.root.attributes('-alpha', 0.6)
        
        self.start_x = None
        self.start_y = None
        self.selection = None

        # O "buraco" é um retângulo preenchido com a cor mágica
        self.rect_fill = self.canvas.create_rectangle(0, 0, 0, 0, fill=TRANS_COLOR, outline="")
        # A borda é um retângulo separado para ser opaca e grossa
        self.rect_border = self.canvas.create_rectangle(0, 0, 0, 0, outline='red', width=2)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_move_press(self, event):
        cur_x, cur_y = (event.x, event.y)
        x1, x2 = sorted([self.start_x, cur_x])
        y1, y2 = sorted([self.start_y, cur_y])
        
        # Atualiza o preenchimento (que será transparente para o Windows)
        self.canvas.coords(self.rect_fill, x1, y1, x2, y2)
        # Atualiza a borda vermelha
        self.canvas.coords(self.rect_border, x1, y1, x2, y2)

    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        # Normaliza coordenadas (caso arraste para trás)
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        # Coordenadas relativas ao Desktop Virtual
        self.selection = (
            int(x1 + self.left), 
            int(y1 + self.top), 
            int(x2 - x1), 
            int(y2 - y1)
        )
        self.root.destroy()

def interactive_capture():
    # 1. Abrir interface de seleção
    selector = RegionSelector()
    selector.root.mainloop()
    
    if not selector.selection or selector.selection[2] < 5:
        return None

    # 2. Capturar a área selecionada usando mss
    x, y, w, h = selector.selection
    with mss.mss() as sct:
        # Nota: mss usa coordenadas globais
        monitor = {"top": y, "left": x, "width": w, "height": h}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Gerar nome aleatório que não existe na pasta
    save_dir = Path(os.getenv('APPDATA')) / "nazm" / "templates"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    random_name = f"cap_{uuid.uuid4().hex[:8]}.png"
    full_path = save_dir / random_name
    
    cv2.imwrite(str(full_path), img)
    return random_name

def list_templates():
    """
    Lista todos os arquivos de imagem salvos na pasta de templates do AppData.
    Retorna uma lista de objetos Path para cada imagem encontrada.
    """
    save_dir = Path(os.getenv('APPDATA')) / "nazm" / "templates"
    
    # Se a pasta nem existir ainda, retorna lista vazia
    if not save_dir.exists():
        return []
    
    # Filtra apenas por extensões de imagem comuns
    extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    templates = [
        p for p in save_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in extensions
    ]
    
    return templates

def list_template_names():
    return [p.name for p in list_templates()]