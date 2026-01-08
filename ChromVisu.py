import tkinter as tk
from tkinter import ttk, messagebox
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AddCompoundWindow:
    def __init__(self, parent, callback):
        self.parent = parent
        self.callback = callback
        self.window = tk.Toplevel(parent)
        self.window.title("Add compound")
        self.window.geometry("400x300")
        
        # Configure le style
        self.window.configure(bg='#f0f0f0')
        
        # Création des champs
        tk.Label(self.window, text="Name:", bg='#f0f0f0', font=('Arial', 10)).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.nom_entry = tk.Entry(self.window, width=30, font=('Arial', 10))
        self.nom_entry.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Label(self.window, text="KD:", bg='#f0f0f0', font=('Arial', 10)).grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.kd_entry = tk.Entry(self.window, width=30, font=('Arial', 10))
        self.kd_entry.grid(row=1, column=1, padx=10, pady=10)
        
        tk.Label(self.window, text="Smiles:", bg='#f0f0f0', font=('Arial', 10)).grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.smiles_entry = tk.Entry(self.window, width=30, font=('Arial', 10))
        self.smiles_entry.grid(row=2, column=1, padx=10, pady=10)
        
        # Bouton d'ajout
        add_button = tk.Button(self.window, text="Add", command=self.add_compound, 
                              bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                              padx=20, pady=5)
        add_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Message d'erreur
        self.error_label = tk.Label(self.window, text="", fg='red', bg='#f0f0f0')
        self.error_label.grid(row=4, column=0, columnspan=2)
        
    def add_compound(self):
        nom = self.nom_entry.get()
        kd = self.kd_entry.get()
        smiles = self.smiles_entry.get()
        
        # Validation
        if not nom or not kd or not smiles:
            self.error_label.config(text="All fields must be completed.")
            return
            
        try:
            kd_value = float(kd)
        except ValueError:
            self.error_label.config(text="KD must be a number")
            return
            
        # Vérification du SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.error_label.config(text="Invalid SMILES")
            return
            
        # Appel du callback pour ajouter au tableau principal
        self.callback(nom, kd_value, smiles)
        self.window.destroy()

class ChromatogramWindow:
    def __init__(self, parent, compounds_data, Vflow, Vm, Voff):
        self.window = tk.Toplevel(parent)
        self.window.title("Chromatogram")
        self.window.geometry("1200x700")
        
        self.compounds_data = compounds_data
        self.Vflow = Vflow
        self.Vm = Vm
        self.Voff = Voff
        
        # Calcul du temps dead volume
        self.Tdv = (Vm + Voff) / Vflow if Vflow > 0 else 0
        
        # Créer la figure matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor('#f5f5f5')
        self.ax.set_facecolor('#fafafa')
        
        # Générer le chromatogramme
        self.generate_chromatogram()
        
        # Configuration de l'axe
        self.ax.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        self.ax.set_title('Predicted chromatogram', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Ajuster les limites
        if hasattr(self, 'time_range'):
            self.ax.set_xlim(0, max(self.time_range) * 1.1)
            # Pour une distribution, normaliser l'échelle Y
            max_density = 0
            for data in compounds_data:
                # Densité maximale = 1/(σ√(2π))
                density_max = 1.0 / (data['sigma_min_t'] * np.sqrt(2 * np.pi))
                max_density = max(max_density, density_max * 1.2)
            self.ax.set_ylim(0, max_density)
        
        # Créer le canvas Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bouton pour fermer
        close_button = tk.Button(self.window, text="Close", command=self.window.destroy,
                                bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                padx=20, pady=5)
        close_button.pack(pady=10)
    
    def gaussian_pdf(self, x, mu, sigma):
        """Distribution gaussienne normalisée (PDF)"""
        # PDF: f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
        # Intégrale = 1
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    
    def generate_chromatogram(self):
        """Génère le chromatogramme avec les distributions gaussiennes"""
        if not self.compounds_data:
            return
            
        # Déterminer la plage de temps
        max_t_retention = max([data['t_retention'] for data in self.compounds_data])
        
        # Déterminer le pic le plus large pour définir la plage de temps
        max_sigma = 0
        for data in self.compounds_data:
            # Calculer sigma pour Nmax (le plus large)
            sigma_max = data['sigma_max_t']
            max_sigma = max(max_sigma, sigma_max)
        
        # Plage de temps : ±4 sigma autour du dernier pic
        self.time_range = np.linspace(0, max_t_retention + 6 * max_sigma, 2000)
        
        # Couleurs pour les composés
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.compounds_data)))
        
        # AJOUT: Tracer la ligne du dead volume en premier (pour qu'elle soit en arrière-plan)
        if self.Tdv > 0:
            # Ligne verticale rouge pour le dead volume
            self.ax.axvline(x=self.Tdv, color='red', linewidth=3, linestyle='-', alpha=0.7, 
                           label=f'Dead volume: Tdv = {self.Tdv:.2f} min')
            
            # Zone hachurée avant le dead volume
            self.ax.axvspan(0, self.Tdv, alpha=0.1, color='red', hatch='//')
            
            # Annotation pour le dead volume
            max_density = max([1.0/(d['sigma_min_t']*np.sqrt(2*np.pi)) for d in self.compounds_data])
            self.ax.annotate(f'Dead volume = {self.Tdv:.2f} min',
                           xy=(self.Tdv, max_density * 0.8),
                           xytext=(self.Tdv * 0.1, max_density * 0.9),
                           arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7),
                           fontsize=10, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='red'))
        
        for i, data in enumerate(self.compounds_data):
            color = colors[i]
            
            # Calculer les PDFs gaussiennes normalisées
            pdf_max = self.gaussian_pdf(self.time_range, data['t_retention'], data['sigma_max_t'])
            pdf_min = self.gaussian_pdf(self.time_range, data['t_retention'], data['sigma_min_t'])
            
            # Calculer les hauteurs maximales (au centre)
            height_max = 1.0 / (data['sigma_max_t'] * np.sqrt(2 * np.pi))
            height_min = 1.0 / (data['sigma_min_t'] * np.sqrt(2 * np.pi))
            
            print(f"Composé {data['letter']}: t={data['t_retention']:.2f} min")
            print(f"  Nmax={data['Nmax']:.0f}: σ={data['sigma_max_t']:.3f} min, hauteur={height_max:.3f}")
            print(f"  Nmin={data['Nmin']:.0f}: σ={data['sigma_min_t']:.3f} min, hauteur={height_min:.3f}")
            print(f"  Ratio hauteur = {height_min/height_max:.1f}")
            
            # Vérifier si le pic est avant le dead volume (ce qui serait anormal)
            if data['t_retention'] < self.Tdv:
                print(f"  WARNING: Peak before dead volume! t_retention={data['t_retention']:.2f} < Tdv={self.Tdv:.2f}")
                # Mettre en évidence ce pic
                color = 'orange'  # Changer la couleur pour alerter
            
            # Tracer la distribution Nmin (trait plein) - plus étroite et plus haute
            self.ax.plot(self.time_range, pdf_min, 
                        color=color, linewidth=2, 
                        label=f"{data['letter']} - Nmin={data['Nmin']:.0f}",
                        solid_capstyle='round')
            
            # Tracer la distribution Nmax (pointillé) - plus large et plus basse
            self.ax.plot(self.time_range, pdf_max, 
                        color=color, linewidth=2, linestyle='--',
                        label=f"{data['letter']} - Nmax={data['Nmax']:.0f}",
                        alpha=0.7)
            
            # Ajouter le texte avec la lettre et le nom
            text_y = height_min * 1.1  # Au-dessus du pic Nmin
            self.ax.text(data['t_retention'], text_y, 
                        f"{data['letter']}: {data['nom']}\nt={data['t_retention']:.1f} min",
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            # Ajouter une ligne verticale au temps de rétention
            self.ax.axvline(x=data['t_retention'], color=color, alpha=0.3, linestyle=':', linewidth=1)
            
            # Afficher les largeurs à mi-hauteur
            half_height_max = height_max / 2
            half_height_min = height_min / 2
            
            # Calculer la largeur à mi-hauteur (FWHM = 2.355 * σ)
            fwhm_max = 2.355 * data['sigma_max_t']
            fwhm_min = 2.355 * data['sigma_min_t']
            
            # Pour Nmax (large et bas)
            self.ax.annotate('', 
                          xy=(data['t_retention'] - fwhm_max/2, half_height_max),
                          xytext=(data['t_retention'] + fwhm_max/2, half_height_max),
                          arrowprops=dict(arrowstyle='<->', color=color, alpha=0.5, linestyle='--'))
            
            # Pour Nmin (étroit et haut)
            self.ax.annotate('', 
                          xy=(data['t_retention'] - fwhm_min/2, half_height_min),
                          xytext=(data['t_retention'] + fwhm_min/2, half_height_min),
                          arrowprops=dict(arrowstyle='<->', color=color, alpha=0.5))
            
            # Afficher les valeurs σ
            self.ax.text(data['t_retention'] + data['sigma_max_t'], height_max * 0.3,
                        f"σ={data['sigma_max_t']:.2f} min\nN={data['Nmax']:.0f}",
                        fontsize=8, color=color, alpha=0.7, ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))
            
            self.ax.text(data['t_retention'] + data['sigma_min_t'], height_min * 0.3,
                        f"σ={data['sigma_min_t']:.2f} min\nN={data['Nmin']:.0f}",
                        fontsize=8, color=color, alpha=0.7, ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

class ChromatogramVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualisation du chromatogramme")
        self.root.geometry("1400x800")
        
        # Style
        self.root.configure(bg='#f5f5f5')
        
        # Configuration du style pour ttk
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables pour stocker les données
        self.compounds = []  # Liste des composés
        self.image_refs = []  # Références aux images pour éviter le garbage collection
        
        # Cadre principal
        main_frame = tk.Frame(root, bg='#f5f5f5')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Titre
        title_label = tk.Label(main_frame, text="Chromatogram visualization", 
                              font=('Arial', 16, 'bold'), bg='#f5f5f5')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Section des paramètres
        params_frame = tk.LabelFrame(main_frame, text="Colonne parameters", 
                                    font=('Arial', 12, 'bold'), bg='#f5f5f5', padx=15, pady=15)
        params_frame.grid(row=1, column=0, sticky='nw', padx=(0, 20), pady=(0, 20))
        
        # Paramètres avec labels et entries
        params = [
            ("Column volume Vcol (mL):", "col_volume"),
            ("Flow Vflow (mL/min):", "debit"),
            ("Volume of stationary phase at equilibrium Vstat (mL):", "phase_stationnaire"),
            ("Injection loop volume Voff (mL):", "injection_loop")
        ]
        
        self.entries = {}
        for i, (label_text, key) in enumerate(params):
            tk.Label(params_frame, text=label_text, bg='#f5f5f5', 
                    font=('Arial', 10)).grid(row=i, column=0, sticky='w', pady=5)
            entry = tk.Entry(params_frame, width=20, font=('Arial', 10))
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries[key] = entry
        
        # Bouton Draw Chromatogram
        self.draw_button = tk.Button(params_frame, text="Draw Chromatogram", 
                                    command=self.calculate_and_display,
                                    bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                    padx=15, pady=8)
        self.draw_button.grid(row=len(params), column=0, columnspan=2, pady=15)
        
        # Section des composés
        compounds_frame = tk.LabelFrame(main_frame, text="Compounds", 
                                       font=('Arial', 12, 'bold'), bg='#f5f5f5', padx=15, pady=15)
        compounds_frame.grid(row=1, column=1, sticky='nsew', padx=(20, 0), pady=(0, 20))
        
        # Boutons pour les composés
        button_frame = tk.Frame(compounds_frame, bg='#f5f5f5')
        button_frame.pack(fill='x', pady=(0, 10))
        
        self.add_button = tk.Button(button_frame, text="Add compound", 
                                   command=self.open_add_window,
                                   bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                   padx=15, pady=8)
        self.add_button.pack(side='left', padx=(0, 10))
        
        self.remove_button = tk.Button(button_frame, text="Delete last compound", 
                                      command=self.remove_selected,
                                      bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                      padx=15, pady=8)
        self.remove_button.pack(side='left')
        
        # Création du tableau avec un Frame personnalisé
        table_frame = tk.Frame(compounds_frame)
        table_frame.pack(fill='both', expand=True)
        
        # Création d'un Canvas pour le tableau
        self.table_canvas = tk.Canvas(table_frame, bg='white')
        self.table_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table_canvas.yview)
        
        # Frame à l'intérieur du Canvas
        self.table_inner_frame = tk.Frame(self.table_canvas, bg='white')
        
        # Configuration du scroll
        self.table_inner_frame.bind(
            "<Configure>",
            lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))
        )
        
        self.table_canvas.create_window((0, 0), window=self.table_inner_frame, anchor="nw")
        self.table_canvas.configure(yscrollcommand=self.table_scrollbar.set)
        
        # En-têtes du tableau
        headers = ['Compound', 'Name', 'KD', 'Structure']
        for col, header in enumerate(headers):
            header_label = tk.Label(self.table_inner_frame, text=header, 
                                   font=('Arial', 10, 'bold'), bg='#e0e0e0',
                                   relief='ridge', padx=10, pady=5, width=20)
            header_label.grid(row=0, column=col, sticky='nsew', padx=1, pady=1)
        
        # Configuration des poids des colonnes
        for i in range(4):
            self.table_inner_frame.columnconfigure(i, weight=1)
        
        self.table_canvas.pack(side='left', fill='both', expand=True)
        self.table_scrollbar.pack(side='right', fill='y')
        
        # Cadre pour les résultats
        self.results_frame = tk.LabelFrame(main_frame, text="Résultats des calculs", 
                                          font=('Arial', 12, 'bold'), bg='#f5f5f5', padx=15, pady=15)
        self.results_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(20, 0))
        
        # Canvas pour les résultats
        self.results_canvas = tk.Canvas(self.results_frame, bg='white', height=250)
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient='vertical', command=self.results_canvas.yview)
        self.results_scrollable_frame = tk.Frame(self.results_canvas, bg='white')
        
        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)
        
        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")
        
        # Label initial pour les résultats
        self.initial_label = tk.Label(self.results_scrollable_frame, 
                                     text="Cliquez sur 'Draw Chromatogram' pour calculer les résultats",
                                     bg='white', font=('Arial', 10, 'italic'), padx=10, pady=10)
        self.initial_label.pack()
        
        # Configuration du redimensionnement
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=0)
    
    def open_add_window(self):
        AddCompoundWindow(self.root, self.add_to_table)
    
    def add_to_table(self, nom, kd, smiles):
        """Ajoute un composé au tableau"""
        try:
            # Générer l'image de la molécule
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Générer l'image de la molécule
                img = Draw.MolToImage(mol, size=(120, 80))
                img_tk = ImageTk.PhotoImage(img)
                
                # Stocker la référence pour éviter le garbage collection
                self.image_refs.append(img_tk)
                
                # Stocker les données du composé
                letter = chr(65 + len(self.compounds))  # A, B, C, ...
                compound_data = {
                    'letter': letter,
                    'nom': nom,
                    'kd': kd,
                    'smiles': smiles,
                    'image': img_tk
                }
                self.compounds.append(compound_data)
                
                # Mettre à jour l'affichage du tableau
                self.update_table_display()
                
                return True
            else:
                messagebox.showerror("Erreur", "SMILES invalide - impossible de générer la molécule")
                return False
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ajout du composé: {str(e)}")
            return False
    
    def update_table_display(self):
        """Met à jour l'affichage du tableau"""
        # Effacer tout le contenu actuel (sauf les en-têtes)
        for widget in self.table_inner_frame.winfo_children():
            if widget.grid_info()['row'] > 0:  # Garder les en-têtes (row=0)
                widget.destroy()
        
        # Afficher tous les composés
        for i, compound in enumerate(self.compounds):
            row = i + 1  # Commencer à la ligne 1 (après les en-têtes)
            
            # Lettre
            letter_label = tk.Label(self.table_inner_frame, text=compound['letter'], 
                                   font=('Arial', 10), bg='white',
                                   relief='ridge', padx=10, pady=5)
            letter_label.grid(row=row, column=0, sticky='nsew', padx=1, pady=1)
            
            # Nom
            nom_label = tk.Label(self.table_inner_frame, text=compound['nom'], 
                                font=('Arial', 10), bg='white',
                                relief='ridge', padx=10, pady=5)
            nom_label.grid(row=row, column=1, sticky='nsew', padx=1, pady=1)
            
            # KD
            kd_label = tk.Label(self.table_inner_frame, text=f"{compound['kd']:.3f}", 
                               font=('Arial', 10), bg='white',
                               relief='ridge', padx=10, pady=5)
            kd_label.grid(row=row, column=2, sticky='nsew', padx=1, pady=1)
            
            # Structure (image)
            structure_label = tk.Label(self.table_inner_frame, 
                                      image=compound['image'],
                                      bg='white', relief='ridge', padx=5, pady=5)
            structure_label.grid(row=row, column=3, sticky='nsew', padx=1, pady=1)
    
    def remove_selected(self):
        """Supprime le composé sélectionné"""
        if not self.compounds:
            return
        
        # Pour simplifier, on supprime le dernier composé ajouté
        if self.compounds:
            # Supprimer le dernier composé
            self.compounds.pop()
            if self.image_refs:
                self.image_refs.pop()
            
            # Mettre à jour l'affichage
            self.update_table_display()
    
    def calculate_and_display(self):
        """Calcule et affiche les résultats"""
        try:
            # Récupérer les valeurs des paramètres
            Vcol = float(self.entries['col_volume'].get())
            Vflow = float(self.entries['debit'].get())
            Vstat = float(self.entries['phase_stationnaire'].get())
            Voff = float(self.entries['injection_loop'].get())
            
            # Calculs de base
            Vm = Vcol - Vstat
            Wmin = Vflow * 1
            Wmax = Vflow * 5
            
            # Calcul du temps dead volume (AJOUT)
            Tdv = (Vm + Voff) / Vflow if Vflow > 0 else 0
            
            # Effacer les résultats précédents
            for widget in self.results_scrollable_frame.winfo_children():
                widget.destroy()
            
            # Afficher les résultats généraux
            general_frame = tk.Frame(self.results_scrollable_frame, bg='white')
            general_frame.pack(fill='x', padx=10, pady=10)
            
            tk.Label(general_frame, text="Paramètres généraux:", 
                    font=('Arial', 11, 'bold'), bg='white').grid(row=0, column=0, sticky='w')
            
            tk.Label(general_frame, text=f"Volume de la phase mobile Vm = Vcol - Vstat = {Vcol:.2f} - {Vstat:.2f} = {Vm:.2f} mL", 
                    bg='white').grid(row=1, column=0, sticky='w', pady=2)
            
            tk.Label(general_frame, text=f"Débit Vflow = {Vflow:.2f} mL/min", 
                    bg='white').grid(row=2, column=0, sticky='w', pady=2)
            
            tk.Label(general_frame, text=f"Volume boucle d'injection Voff = {Voff:.2f} mL", 
                    bg='white').grid(row=3, column=0, sticky='w', pady=2)
            
            # AJOUT: Afficher le calcul du dead volume
            tk.Label(general_frame, text=f"Temps dead volume Tdv = (Vm + Voff) / Vflow = ({Vm:.2f} + {Voff:.2f}) / {Vflow:.2f} = {Tdv:.2f} min", 
                    bg='white', fg='red', font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky='w', pady=2)
            
            tk.Label(general_frame, text=f"Wmin = Vflow * 1 = {Vflow:.2f} * 1 = {Wmin:.2f} mL/min", 
                    bg='white').grid(row=5, column=0, sticky='w', pady=2)
            
            tk.Label(general_frame, text=f"Wmax = Vflow * 5 = {Vflow:.2f} * 5 = {Wmax:.2f} mL/min", 
                    bg='white').grid(row=6, column=0, sticky='w', pady=2)
            
            tk.Label(general_frame, text="", bg='white').grid(row=7, column=0, pady=10)
            
            # Préparer les données pour le chromatogramme
            compounds_data_for_chromatogram = []
            
            # Calculs pour chaque composé
            if self.compounds:
                compounds_frame = tk.Frame(self.results_scrollable_frame, bg='white')
                compounds_frame.pack(fill='x', padx=10, pady=10)
                
                tk.Label(compounds_frame, text="Distributions de temps d'élution:", 
                        font=('Arial', 11, 'bold'), bg='white').grid(row=0, column=0, sticky='w', columnspan=2)
                
                # AJOUT: Avertissement si des pics sont avant le dead volume
                warning_shown = False
                
                # Calculer et afficher Vr pour chaque composé
                vr_values = []
                row_counter = 1
                
                for compound in self.compounds:
                    # Calcul du volume de rétention
                    Vr = (compound['kd'] * Vstat) + Vm + Voff
                    
                    # Calcul du temps de rétention (moyenne de la distribution)
                    t_retention = Vr / Vflow
                    
                    # Vérification si le pic est avant le dead volume (AJOUT)
                    if t_retention < Tdv and not warning_shown:
                        warning_frame = tk.Frame(self.results_scrollable_frame, bg='yellow', padx=10, pady=10)
                        warning_frame.pack(fill='x', padx=10, pady=5)
                        tk.Label(warning_frame, 
                                text=f"ATTENTION: Le composé {compound['letter']} a un temps de rétention ({t_retention:.2f} min) inférieur au dead volume ({Tdv:.2f} min)!",
                                font=('Arial', 10, 'bold'), bg='yellow').pack()
                        warning_shown = True
                    
                    # Calcul des nombres de plateaux théoriques
                    Nmax = 4 * ((Vr / Wmax) ** 2)
                    Nmin = 4 * ((Vr / Wmin) ** 2)
                    
                    # Calcul des écarts-types (sigma en temps)
                    sigma_max_t = t_retention / np.sqrt(Nmax) if Nmax > 0 else 0
                    sigma_min_t = t_retention / np.sqrt(Nmin) if Nmin > 0 else 0
                    
                    # Calcul des hauteurs maximales des PDFs
                    height_max = 1.0 / (sigma_max_t * np.sqrt(2 * np.pi)) if sigma_max_t > 0 else 0
                    height_min = 1.0 / (sigma_min_t * np.sqrt(2 * np.pi)) if sigma_min_t > 0 else 0
                    
                    # Largeur à mi-hauteur (FWHM)
                    fwhm_max = 2.355 * sigma_max_t
                    fwhm_min = 2.355 * sigma_min_t
                    
                    vr_values.append((compound['letter'], Vr, compound['kd'], t_retention, 
                                    Nmax, Nmin, sigma_max_t, sigma_min_t, height_max, height_min, fwhm_max, fwhm_min))
                    
                    # Afficher les résultats pour ce composé
                    tk.Label(compounds_frame, 
                            text=f"{compound['letter']} ({compound['nom']}): Distribution Gaussienne", 
                            font=('Arial', 10, 'bold'), bg='white').grid(row=row_counter, column=0, sticky='w', pady=2)
                    
                    tk.Label(compounds_frame, 
                            text=f"  Paramètres: Vr = {Vr:.2f} mL, t = {t_retention:.2f} min", 
                            bg='white').grid(row=row_counter+1, column=0, sticky='w', padx=20)
                    
                    # AJOUT: Comparaison avec le dead volume
                    if t_retention < Tdv:
                        tk.Label(compounds_frame, 
                                text=f"  ⚠️ t < Tdv ({t_retention:.2f} < {Tdv:.2f} min) - Pic avant le dead volume!", 
                                bg='white', fg='orange', font=('Arial', 9, 'bold')).grid(row=row_counter+2, column=0, sticky='w', padx=20)
                        row_counter += 1
                    else:
                        tk.Label(compounds_frame, 
                                text=f"  ✓ t > Tdv ({t_retention:.2f} > {Tdv:.2f} min)", 
                                bg='white', fg='green', font=('Arial', 9)).grid(row=row_counter+2, column=0, sticky='w', padx=20)
                    
                    tk.Label(compounds_frame, 
                            text="  Distribution Nmax (faible efficacité):", 
                            bg='white', font=('Arial', 9, 'bold')).grid(row=row_counter+3, column=0, sticky='w', padx=20)
                    
                    tk.Label(compounds_frame, 
                            text=f"    N = {Nmax:.0f}, σ = {sigma_max_t:.3f} min, FWHM = {fwhm_max:.3f} min", 
                            bg='white', font=('Arial', 9)).grid(row=row_counter+4, column=0, sticky='w', padx=40)
                    
                    tk.Label(compounds_frame, 
                            text=f"    Hauteur PDF: 1/(σ√(2π)) = {height_max:.3f}", 
                            bg='white', font=('Arial', 9)).grid(row=row_counter+5, column=0, sticky='w', padx=40)
                    
                    tk.Label(compounds_frame, 
                            text="  Distribution Nmin (haute efficacité):", 
                            bg='white', font=('Arial', 9, 'bold')).grid(row=row_counter+6, column=0, sticky='w', padx=20)
                    
                    tk.Label(compounds_frame, 
                            text=f"    N = {Nmin:.0f}, σ = {sigma_min_t:.3f} min, FWHM = {fwhm_min:.3f} min", 
                            bg='white', font=('Arial', 9)).grid(row=row_counter+7, column=0, sticky='w', padx=40)
                    
                    tk.Label(compounds_frame, 
                            text=f"    Hauteur PDF: 1/(σ√(2π)) = {height_min:.3f}", 
                            bg='white', font=('Arial', 9)).grid(row=row_counter+8, column=0, sticky='w', padx=40)
                    
                    tk.Label(compounds_frame, 
                            text=f"  Facteur d'élargissement: σ_max/σ_min = {sigma_max_t/sigma_min_t:.1f}", 
                            bg='white', font=('Arial', 9)).grid(row=row_counter+9, column=0, sticky='w', padx=20)
                    
                    tk.Label(compounds_frame, 
                            text=f"  Facteur hauteur: h_min/h_max = {height_min/height_max:.1f}", 
                            bg='white', font=('Arial', 9)).grid(row=row_counter+10, column=0, sticky='w', padx=20)
                    
                    # Stocker les données pour le chromatogramme
                    compounds_data_for_chromatogram.append({
                        'letter': compound['letter'],
                        'nom': compound['nom'],
                        'Vr': Vr,
                        't_retention': t_retention,
                        'Nmax': Nmax,
                        'Nmin': Nmin,
                        'sigma_max_t': sigma_max_t,
                        'sigma_min_t': sigma_min_t,
                        'height_max': height_max,
                        'height_min': height_min,
                        'fwhm_max': fwhm_max,
                        'fwhm_min': fwhm_min
                    })
                    
                    row_counter += 12
                    tk.Label(compounds_frame, text="", bg='white').grid(row=row_counter, column=0, pady=5)
                    row_counter += 1
                
                # Calculs de sélectivité
                if len(self.compounds) >= 2:
                    selectivity_frame = tk.Frame(self.results_scrollable_frame, bg='white')
                    selectivity_frame.pack(fill='x', padx=10, pady=10)
                    
                    tk.Label(selectivity_frame, text="Sélectivité (α):", 
                            font=('Arial', 11, 'bold'), bg='white').grid(row=0, column=0, sticky='w')
                    
                    row_num = 1
                    # Calculer toutes les paires
                    for i in range(len(vr_values)):
                        for j in range(i+1, len(vr_values)):
                            comp1 = vr_values[i]
                            comp2 = vr_values[j]
                            
                            # Calculer alpha (KD(A)/KD(B))
                            if comp2[2] != 0:  # Éviter la division par zéro
                                alpha = comp1[2] / comp2[2]
                                # Toujours avoir alpha >= 1
                                if alpha < 1:
                                    alpha = 1/alpha
                                    tk.Label(selectivity_frame, 
                                            text=f"α({comp2[0]}/{comp1[0]}) = KD({comp2[0]})/KD({comp1[0]}) = {comp2[2]:.3f}/{comp1[2]:.3f} = {alpha:.3f}", 
                                            bg='white').grid(row=row_num, column=0, sticky='w', pady=2)
                                else:
                                    tk.Label(selectivity_frame, 
                                            text=f"α({comp1[0]}/{comp2[0]}) = KD({comp1[0]})/KD({comp2[0]}) = {comp1[2]:.3f}/{comp2[2]:.3f} = {alpha:.3f}", 
                                            bg='white').grid(row=row_num, column=0, sticky='w', pady=2)
                                row_num += 1
                
                # Ouvrir la fenêtre du chromatogramme
                if compounds_data_for_chromatogram:
                    # AJOUT: Passer Vm et Voff à la fenêtre du chromatogramme
                    ChromatogramWindow(self.root, compounds_data_for_chromatogram, Vflow, Vm, Voff)
            
            else:
                tk.Label(self.results_scrollable_frame, text="Aucun composé dans le tableau. Ajoutez des composés pour effectuer les calculs.", 
                        bg='white', font=('Arial', 10, 'italic')).pack(padx=10, pady=10)
            
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides pour tous les paramètres.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

def main():
    root = tk.Tk()
    app = ChromatogramVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()