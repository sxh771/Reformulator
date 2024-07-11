import numpy as np
import pandas as pd
import itertools
import math
import tkinter.ttk as ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import os
import uuid
import warnings
import json
import copy

from simulator import *

# Not yet implemented - window allowing users to add custom temperature profiles.
class TempProfileEditor(tk.Toplevel):
        def __init__(self):
                super().__init__()
                ttk.Label(self, text="Profile Name: ").grid(row=0,column=0)
                self.pname_entry  = ttk.Entry(self)
                self.pname_entry.grid(row=0,column=1)
                self.var_frame = ttk.Frame(self)
                self.var_frame.grid(row=1,column=0,columnspan=2)
                ttk.Label(self.var_frame, text="Display Name").grid(row=0,column=0)
                ttk.Label(self.var_frame, text="Variable Name").grid(row=0,column=1)
                self.var_entries = []
                self.add_variable()
                ttk.Button(self, text="Add Variable", command = self.add_variable).grid(row=2,column=0)
                ttk.Button(self, text="Autoname", command = self.autoname).grid(row=2,column=1)
                ttk.Label(self, text="Enter Function Below").grid(row=3,column=0)
                self.function_entry = ttk.Entry(self)
                self.function_entry.grid(row=4,column=0, columnspan=2, sticky="NSEW")
                ttk.Button(self, text="Save", command = self.save_profile).grid(row=5,column=0)
                ttk.Button(self, text="Cancel", command = self.destroy).grid(row=5,column=1)
                
        def remove_row(self, row_num):
                for e in var_entries[row-1]:
                        e.destroy()
                del var_entries[row-1]
        
        def add_variable(self):
                display_entry = ttk.Entry(self.var_frame)
                cur_row = 1+len(self.var_entries)
                display_entry.grid(row=cur_row, column=0)
                var_entry = ttk.Entry(self.var_frame)
                var_entry.grid(row=cur_row, column=1)
                rmv_btn = ttk.Button(self.var_frame, text="X", command = lambda: self.remove_row(cur_row))
                rmv_btn.grid(row=cur_row, column=2)
                self.var_entries.append((display_entry, var_entry, rmv_btn))

        def autoname(self):
                for display_entry, var_entry, _ in self.var_entries:
                        display_name = display_entry.get()
                        var_name = display_name.lower()
                        if 0x30 <= ord(var_name[0]) <= 0x39:
                                var_name = "_"+var_name[1:]
                        for sc in [" ", "(",")","[","]",".","@","*","+","-","/","&","^","~","|","{","}","%"]:
                                var_name = var_name.replace(sc, "_")
                        var_entry.delete(0, tk.END)
                        var_entry.insert(0, var_name)

        def save_profile(self):
                pass
        
                
                
                
class TempProfileFrame(ttk.Frame):
        """
        Frame for selecting and entering temperature ramp parameters. Used on main & reformulate screens.
        """
        def __init__(self, root):
                super().__init__(root)
                profile_options = {
                        "Constant": {
                                "vars": {
                                        "temp": "Temperature (C)"
                                }, 
                                "func": "temp"
                        },
                        "Linear Ramp":{
                                "vars":{
                                        "flash_time": "Flash Time (min)", 
                                        "start_temp": "Initial Temperature (C)", 
                                        "end_temp": "Final Temperature (C)", 
                                        "ramp_time": "Ramp Time (min)"
                                }, 
                                "func": "np.clip((t-flash_time)/ramp_time * (end_temp - start_temp) + start_temp, start_temp, end_temp)"
                        },
                        "Exponential Ramp": {
                                "vars":{
                                        "flash_time": "Flash Time (min)", 
                                        "start_temp": "Initial Temperature (C)", 
                                        "end_temp": "Final Temperature (C)", 
                                        "ramp_time": "Ramp Time Constant (min)"
                                },
                                "func": "np.clip(end_temp - np.exp(-(t-flash_time)/ramp_time) * (end_temp - start_temp), start_temp, end_temp)"
                        }
                }
                ttk.Label(self, text="Temperature Profile").grid(row=0, column=0)
                #self.add_new_profile = ttk.Button(self, text="Add New", command=self.add_profile_option)
                #self.add_new_profile.grid(row=0,column=1)
                self.t_profile_selection = tk.StringVar(self)
                self.t_profile_dropdown = None
                self.profile_frames = []
                self.update_profile_options(profile_options)

        def add_profile_option(self):
                
                self.t_profile_dropdown.configure(values=self.t_profile_dropdown.cget("values")+[name])
                new_frame = ttk.Frame(self)
                new_frame.grid(row=len(self.profile_options.keys()),column=0,columnspan=2)
                self.profile_frames.append(new_frame)
                info["entry_map"] = dict()
                for j, var_label in enumerate(profile_options[profile]["vars"].values()):
                        ttk.Label(new_frame, text=var_label).grid(row=j,column=0)
                        info["entry_map"][var_label] = ttk.Entry(new_frame)
                        info["entry_map"][var_label].grid(row=j, column=1)
                self.profile_options[name] = info
                
        def update_profile_options(self, profile_options):
                if self.t_profile_dropdown is not None:
                        self.t_profile_dropdown.destroy()
                self.t_profile_dropdown = ttk.Combobox(self, textvariable=self.t_profile_selection, values=list(profile_options.keys()))
                self.t_profile_dropdown.grid(row=0, column=1)
                self.profile_options = copy.deepcopy(profile_options)
                
                for i, profile in enumerate(profile_options.keys()):
                        self.profile_options[profile]["index"] = i
                        cur_frame = ttk.Frame(self)
                        cur_frame.grid(row=i+1,column=0,columnspan=2)
                        self.profile_frames.append(cur_frame)
                        self.profile_options[profile]["entry_map"] = dict()
                        for j, var_label in enumerate(profile_options[profile]["vars"].values()):
                                ttk.Label(cur_frame, text=var_label).grid(row=j,column=0)
                                self.profile_options[profile]["entry_map"][var_label] = ttk.Entry(cur_frame)
                                self.profile_options[profile]["entry_map"][var_label].grid(row=j, column=1)
                self.t_profile_selection.trace("w",self.update_temp_entry)
                self.t_profile_selection.set(list(self.profile_options.keys())[0])
                
        def update_temp_entry(self, *args):
                show = self.profile_options[self.t_profile_selection.get()]["index"]
                for i, frame in enumerate(self.profile_frames):
                        if i == show:
                                frame.grid()
                        else:
                                frame.grid_remove()

        def get_temp_profile(self):
                try:
                        cur_profile = self.profile_options[self.t_profile_selection.get()]
                except KeyError:
                        tk.messagebox.showwarning(title="Invalid Input", message="Invalid temperature profile name")
                        return
                try:
                        [float(e.get()) for e in cur_profile["entry_map"].values()]
                except ValueError:
                        tk.messagebox.showwarning(title="Invalid Input", message="Please enter decimal numbers for temperature curve parameters")
                        return
                try:
                        t = np.arange(0, 10)
                        exec("\n".join([f"{var}={cur_profile['entry_map'][var_label].get()}" for var, var_label in cur_profile["vars"].items()]))
                        eval(cur_profile["func"])
                except Exception:
                        tk.messagebox.showwarning(title="Invalid Input", message="Invalid temperature curve function/parameter combination")
                        return 
                def temp_curve(t):
                        exec("\n".join([f"{var}={cur_profile['entry_map'][var_label].get()}" for var, var_label in cur_profile["vars"].items()]))
                        return eval(cur_profile["func"])
                return temp_curve

        def get_config(self):
                ret = {k:copy.copy(v) for k, v in self.profile_options.items()}
                for profile in ret.values():
                        profile["entry_map"] = {k:v.get() for k, v in profile["entry_map"].items()}
                return ret
        
        def set_config(self, config):
                for name in config:
                        if name in self.profile_options:
                                for variable, entry in self.profile_options[name]["entry_map"].items():
                                        if variable in config[name]["entry_map"]:
                                                entry.delete(0, tk.END)
                                                entry.insert(0, config[name]["entry_map"][variable])
                                                

class TargetParamFrame(ttk.Frame):
        """
        Frame for entering target resin parameters. Used on main & reformulate screens.
        """
        def __init__(self, root):
                super().__init__(root)
                ttk.Label(self, text="Target Parameters", font=("Arial", 14)).grid(row=4, column=0, columnspan=2)
                
                ttk.Label(self, text="δD").grid(row=5,column=0)
                self.dD_entry = ttk.Entry(self)
                self.dD_entry.grid(row=5, column=1)

                ttk.Label(self, text="δP").grid(row=6,column=0)
                self.dP_entry = ttk.Entry(self)
                self.dP_entry.grid(row=6, column=1)

                ttk.Label(self, text="δH").grid(row=7,column=0)
                self.dH_entry = ttk.Entry(self)
                self.dH_entry.grid(row=7, column=1)

                ttk.Label(self, text="R").grid(row=8,column=0)
                self.R_entry = ttk.Entry(self)
                self.R_entry.grid(row=8, column=1)

                ttk.Label(self, text="Wt% Solids").grid(row=9,column=0)
                self.solids_entry = ttk.Entry(self)
                self.solids_entry.grid(row=9, column=1)
                
        def get_params(self):
                try:
                        assert float(self.R_entry.get()) > 0 and 0 <= float(self.solids_entry.get()) <= 100
                        return {"dD":float(self.dD_entry.get()), "dP":float(self.dP_entry.get()), "dH":float(self.dH_entry.get()), "R0": float(self.R_entry.get()), "solids": float(self.solids_entry.get())}
                except ValueError:
                        tk.messagebox.showwarning(title="Invalid Input", message="Please enter decimal numbers for target parameters (R > 0, 0 <= solids <= 100)")
                        return
        def get_config(self):
                return {"dD":self.dD_entry.get(), "dP":self.dP_entry.get(), "dH":self.dH_entry.get(), "R0": self.R_entry.get(), "solids": self.solids_entry.get()}
        def set_config(self, config):
                for entry, value in zip([self.dD_entry, self.dP_entry, self.dH_entry, self.R_entry, self.solids_entry], config.values()):
                        entry.delete(0, tk.END)
                        entry.insert(0, value)
        

                

class MainInputFrame(ttk.Frame):
        """
        Handles all input for the main program screen.
        """
        def __init__(self, root, graph, compare_input, compare_screen, reform_input, reform_screen):
                super().__init__(root)
                self.graph = graph
                self.compare_input = compare_input
                self.compare_screen = compare_screen
                self.reform_input = reform_input
                self.reform_input.set_main_input(self)
                self.reform_screen = reform_screen
                ttk.Label(self, text="Formulation", font=("Arial", 14)).grid(row=0, column=0, columnspan=2)

                self.solvent_button = ttk.Button(self, text="Select Solvent Info File", command=self.select_solvent_file)
                self.solvent_label = ttk.Label(self, text="")
                self.solvent_fname = ""
                self.solvent_button.grid(row=1, column=0)
                self.solvent_label.grid(row=1, column=1)
                
                self.formulation_button = ttk.Button(self, text="Select Formulation File", command=self.select_formulation_file)
                self.formulation_label = ttk.Label(self, text="")
                self.formulation_button.grid(row=2, column=0)
                self.formulation_label.grid(row=2, column=1)
                self.formulation_fname = ""

                self.formulation_table = ttk.Treeview(self, columns=('chemical_name','conc'), show="headings")
                self.formulation_table.heading('chemical_name', text='Chemical Name')
                self.formulation_table.heading('conc', text='Weight Fraction')
                self.formulation_table.grid(row=3, column=0, columnspan=2)
                
                self.target_frame = TargetParamFrame(self)
                self.target_frame.grid(row=4, column=0, columnspan=2, rowspan=5)
                
                self.temp_frame = TempProfileFrame(self)
                self.temp_frame.grid(row=9, column=0, columnspan=2, rowspan=3)

                ttk.Label(self, text="Total Time (min): ").grid(row=12,column=0)
                self.total_time = ttk.Entry(self)
                self.total_time.grid(row=12,column=1)
                self.big_style = ttk.Style()
                self.big_style.configure("big.TButton", font=('Arial',14))
                ttk.Button(self, text="Run Simulation", style='big.TButton', command=self.run_evap_predictor).grid(row=13,column=0,columnspan=2)
                ttk.Button(self, text="Export to Excel", style='big.TButton', command=self.write_output).grid(row=14,column=0,columnspan=2)
                ttk.Button(self, text="Compare Outputs", style='big.TButton', command=self.compare_screen.deiconify).grid(row=15,column=0,columnspan=2)
                ttk.Button(self, text="Reformulate", style='big.TButton', command=self.reform_screen.deiconify).grid(row=16,column=0,columnspan=2)
                ttk.Button(self, text="Load Config", style='big.TButton', command=self.read_config).grid(row=17,column=0)
                ttk.Button(self, text="Save Config", style='big.TButton', command=self.write_config).grid(row=17,column=1)
                self.all_solvents_df = None
                
        def select_formulation_file(self, fname=None):
                if fname is None:
                        fname = tk.filedialog.askopenfilename(filetypes=[("Excel files  (.xlsx)", "*.xlsx")])
                self.formulation_fname = fname
                try:
                        self.formulation_df = pd.read_excel(fname)
                except Exception:
                        if fname == "":
                                return
                        tk.messagebox.showwarning(title="Could not Open", message=f"Could not open file {fname}, likely don't have read permission (often happens if the file is in onedrive)")
                        return
                if "Name" not in self.formulation_df.columns or ("Volume Fraction" not in self.formulation_df.columns and "Weight Fraction" not in self.formulation_df.columns):
                        tk.messagebox.showwarning(title="Invalid Input", message="Formulation file missing either name column or both weight and volume fraction column")
                        self.focus_force()
                        self.formulation_fname = ""
                        self.formulation_df = None
                self.convert_to_weight = "Volume Fraction" not in self.formulation_df.columns
                try:
                        if self.convert_to_weight:
                                self.formulation_table.heading('conc', text='Weight Fraction')
                                for index, row in self.formulation_df.iterrows():
                                        assert type(row["Weight Fraction"]) in [int, float]
                                        self.formulation_table.insert('', tk.END, values=(row["Name"], row["Weight Fraction"]))
                        else:
                                self.formulation_table.heading('conc', text='Volume Fraction')
                                for index, row in self.formulation_df.iterrows():
                                        assert type(row["Volume Fraction"]) in [int, float]
                                        self.formulation_table.insert('', tk.END, values=(row["Name"], row["Volume Fraction"]))
                except AssertionError:
                        tk.messagebox.showwarning(title="Invalid Input", message="Formulation file has invalid data")
                        self.focus_force()
                        self.formulation_fname = ""
                        self.formulation_df = None
                        return
                self.formulation_label.configure(text=os.path.basename(fname))
                
        def select_solvent_file(self, fname=None):
                if fname is None:
                        fname = tk.filedialog.askopenfilename(filetypes=[("Excel files  (.xlsx)", "*.xlsx")])
                try:
                        self.all_solvents_df = pd.read_excel(fname, index_col="Solvents")
                except Exception:
                        if fname == "":
                                return
                        tk.messagebox.showwarning(title="Could not Open", message=f"Could not open file {fname}, likely don't have read permission (often happens if the file is in onedrive)")
                        return
                #Check file format 
                num_cols = ['δD', 'δP', 'δH', 'RER', 'AA', 'AB', 'AC', 'VP@25', 'Density']
                bool_cols = ['Exempt', 'Multi-Component']
                other_cols = ['Components', 'Amounts']
                reqd_cols = num_cols + bool_cols + other_cols
                if not all([x in self.all_solvents_df.columns for x in reqd_cols]):
                        tk.messagebox.showwarning(title="Invalid Input", message=f"Solvent info file missing required columns {' '.join([x for x in reqd_cols if x not in self.all_solvents_df.columns])}")
                        self.focus_force()
                        self.all_solvents_df = None
                        return
                for col in num_cols:
                        try:
                                assert all([x == "-" or str(x).replace('.','',1).isdigit() for x in self.all_solvents_df[col]])
                        except Exception:
                                print([x for x in self.all_solvents_df[col]])
                                print([x == "-" or str(x).replace('.','',1).isdigit() for x in self.all_solvents_df[col]])
                                tk.messagebox.showwarning(title="Invalid Input", message=f"Bad data in column {col} of solvent info")
                                self.focus_force()
                                self.all_solvents_df = None
                                return
                for col in bool_cols:
                        try:
                                assert all([type(x) == bool for x in self.all_solvents_df[col]])
                        except Exception:
                                tk.messagebox.showwarning(title="Invalid Input", message=f"Bad data in column {col} of solvent info")
                                self.focus_force()
                                self.all_solvents_df = None
                                return
                self.solvent_fname = fname
                self.solvent_label.configure(text=os.path.basename(fname))
                self.reform_input.solvent_label.configure(text=os.path.basename(fname))

        def run_evap_predictor(self):
                """
                Processes inputs and sends them to :func:`get_evap_curve`"
                """
                if "Volume Fraction" not in self.formulation_df.columns:
                        volume = self.formulation_df["Weight Fraction"] / np.array(self.all_solvents_df["Density"][self.formulation_df["Name"]])
                        self.formulation_df["Volume Fraction"] = volume / sum(volume)
                try:
                        blend = [self.all_solvents_df.loc[name,:] for name in self.formulation_df["Name"]]
                except KeyError:
                        tk.messagebox.showwarning(title="Invalid Input", message="Name mismatch between formulation and solvent database")
                        return
                try:
                        conc = self.formulation_df["Volume Fraction"]/sum(self.formulation_df["Volume Fraction"])
                        assert not any(np.isnan(np.array(conc)))
                except Exception:
                        tk.messagebox.showwarning(title="Invalid Input", message="Volume fractions could not be computed - were densities entered correctly?")
                        return
                try:
                        t_span = [0, float(self.total_time.get())]
                except ValueError:
                        tk.messagebox.showwarning(title="Invalid Input", message="Please enter a decimal number for total time")
                        return
                target = self.target_frame.get_params()
                if target is None:
                        return
                temp_curve = self.temp_frame.get_temp_profile()
                if temp_curve is None:
                        return
                self.t, self.total_profile, self.partial_profiles, self.RED = get_evap_curve(conc, blend, target, temp_curve, t_span,
                                                                                             self.all_solvents_df, self.convert_to_weight)
                self.temp = [temp_curve(ts) for ts in self.t]
                self.graph.clear_artists()
                self.graph.temp_ax.plot(self.t, self.temp)
                self.graph.temp_ax.relim()
                
                self.graph.RED_ax.plot(self.t, self.RED)
                self.graph.RED_ax.relim()
                
                for pp, name in zip(self.partial_profiles, self.formulation_df["Name"]):
                        self.graph.conc_ax.plot(self.t, pp, label = name)
                self.graph.conc_ax.set_ylabel("Weight Fraction" if self.convert_to_weight else "Volume Fraction")
                self.graph.conc_ax.plot(self.t, self.total_profile, label="Total")
                self.graph.conc_ax.relim()
                self.graph.conc_ax.legend(fontsize=6)
                self.graph.canvas.draw()

        def write_output(self):
                fname = tk.filedialog.asksaveasfilename(filetypes=[("Excel files (.xlsx)", "*.xlsx")])
                if fname[-5:] != ".xlsx":
                        fname = fname + ".xlsx"
                write_to_excel(fname, self.formulation_df["Name"], self.t, self.total_profile,
                               self.partial_profiles, self.RED, self.temp, "Weight Fraction" if self.convert_to_weight else "Volume Fraction")
                os.startfile(fname)

        def get_config(self):
                return {"temp_frame": self.temp_frame.get_config(),
                        "target_frame": self.target_frame.get_config(),
                        "formulation_fname":self.formulation_fname,
                        "solvent_fname":self.solvent_fname,
                        "total_time": self.total_time.get(),
                        "compare_input": self.compare_input.get_config(),
                        "reform_input":self.reform_input.get_config()}
        
        def set_config(self, config):
                self.compare_input.set_config(config["compare_input"])
                self.reform_input.set_config(config["reform_input"])
                self.temp_frame.set_config(config["temp_frame"])
                self.target_frame.set_config(config["target_frame"])
                if config["formulation_fname"] != "":
                        self.select_formulation_file(config["formulation_fname"])
                if config["solvent_fname"] != "":
                        self.select_solvent_file(config["solvent_fname"])
                self.total_time.delete(0, tk.END)
                self.total_time.insert(0, config["total_time"])

        def write_config(self):
                fname = tk.filedialog.asksaveasfilename(filetypes=[("JSON files (.json)", "*.json")])
                if fname[-5:] != ".json":
                        fname += ".json"
                with open(fname, "w") as f:
                        json.dump(self.get_config(), f)
        def read_config(self):
                fname = tk.filedialog.askopenfilename(filetypes=[("JSON files (.json)", "*.json")])
                with open(fname) as f:
                        self.set_config(json.load(f))
                
                

class CompareInputFrame(ttk.Frame):
        """
        Handles all input for the comparison screen
        """
        def __init__(self, root, graph):
                super().__init__(root)
                self.graph = graph
                self.files = set()
                self.file_buttons = dict()
                ttk.Button(self, text="Select Output Files", command=self.get_files).grid(row=0,column=0)
                self.file_frame = ttk.Frame(self)
                self.file_frame.grid(row=1,column=0)
                self.file_frame_row = 0
                self.big_style = ttk.Style()
                self.big_style.configure("big.TButton", font=('Arial',14))
                ttk.Label(self, text="Compare:").grid(row=2, column=0)
                self.plot_total = tk.IntVar()
                self.plot_partial = tk.IntVar()
                self.plot_RED = tk.IntVar()
                self.plot_temp = tk.IntVar()
                ttk.Checkbutton(self, text="Total Profiles", variable=self.plot_total).grid(row=3, column=0, columnspan=2)
                ttk.Checkbutton(self, text="Partial Profiles", variable=self.plot_partial).grid(row=4, column=0, columnspan=2)
                ttk.Checkbutton(self, text="RED", variable=self.plot_RED).grid(row=5, column=0, columnspan=2)
                ttk.Checkbutton(self, text="Temperature Profiles", variable=self.plot_temp).grid(row=6, column=0, columnspan=2)
                ttk.Button(self, text="Compare", style="big.TButton", command=self.compare).grid(row=7, column=0, columnspan=2)
                self.whitelist = []
                self.blacklist = []
                self.using_whitelist = False
                
        def remove_file(self, fname):
                for w in self.file_buttons[fname]:
                        w.grid_forget()
                del self.file_buttons[fname]
                self.files.remove(fname)
                
        def add_files(self, new_full_files):
                self.focus_force()
                new_files = [os.path.basename(f) for f in new_full_files]
                for base, full in zip(new_files, new_full_files):
                        if full not in self.files:
                                file_name = ttk.Label(self.file_frame, text=base)
                                file_name.grid(row=self.file_frame_row, column=0)
                                x_button = ttk.Button(self.file_frame, text="X", command=lambda f=full:self.remove_file(f))
                                x_button.grid(row=self.file_frame_row, column=1)
                                self.file_frame_row += 1
                                self.file_buttons[full] = [file_name, x_button]
                                
                self.files = self.files.union(new_full_files)

        def get_files(self):
                self.add_files(tk.filedialog.askopenfilenames())
                
        def clear(self):
                for fname in self.files:
                        for w in self.file_buttons[fname]:
                                w.grid_forget()
                        del self.file_buttons[fname]
                self.files = set()
                self.graph.clear_artists()
                
        def compare(self):
                """
                Generate comparison plot using loaded data and inputs.
                """
                self.graph.clear_artists()
                mode = None
                for fname in self.files:
                        base = os.path.basename(fname)
                        xl = pd.ExcelFile(fname)
                        if len(xl.sheet_names) != 2:
                                tk.messagebox.showwarning(title="Invalid Input", message=f"Bad format - {fname}")
                                return
                        if mode == None:
                                mode = xl.sheet_names[1]
                        elif xl.sheet_names[1] != mode:
                                tk.messagebox.showwarning(title="Invalid Input", message=f"Can't mix weight% and volume% - {fname}")
                                return
                        df = xl.parse(mode)
                        try:
                                if self.plot_total.get():
                                        self.graph.conc_ax.plot(df.iloc[:,1], df.iloc[:,-3], label=base)
                                if self.plot_partial.get():
                                        for i in range(3, len(df.columns)-3):
                                                self.graph.conc_ax.plot(df.iloc[:,1], df.iloc[:,i:i+1], label=f"{base} - {df.columns[i]}")
                                if self.plot_RED.get():
                                        self.graph.RED_ax.plot(df.iloc[:,1], df.iloc[:,-2], label=base)
                                if self.plot_temp.get():
                                        self.graph.temp_ax.plot(df.iloc[:,1], df.iloc[:,-1], label=base)
                        except Exception:
                                tk.messagebox.showwarning(title="Invalid Input", message=f"Bad format - {fname}")
                                return
                self.graph.conc_ax.set_ylabel(mode[10:-1])
                self.graph.temp_ax.relim()
                self.graph.RED_ax.relim()
                self.graph.conc_ax.relim()
                self.graph.conc_ax.legend(fontsize=6)
                self.graph.canvas.draw()
                
        def get_config(self):
                return {"files": list(self.files)}
        def set_config(self, config):
                self.clear()
                self.add_files(config["files"])

class ReformInputFrame(ttk.Frame):
        """
        Handles all input for the reformulation/alternative solvent selection screen.
        """
        def __init__(self, root, compare_screen, compare_input, results_frame):
                super().__init__(root)
                self.compare_screen = compare_screen
                self.compare_input = compare_input
                self.results_frame = results_frame
                ttk.Button(self, text="Load Control Blend", command=self.load_control_blend).grid(row=0,column=0,columnspan=2)
                self.control_blend_label = ttk.Label(self, text="")
                self.control_blend_label.grid(row=0,column=2,columnspan=2)
                ttk.Button(self, text="Load Minimum Composition", command=self.load_min_comp).grid(row=1,column=0,columnspan=2)
                self.min_comp_label = ttk.Label(self, text="")
                self.min_comp_label.grid(row=1,column=2,columnspan=2)
                ttk.Button(self, text="Load Whitelist/Blacklist", command=self.load_wl_bl).grid(row=2,column=0,columnspan=2)
                self.wl_bl_label = ttk.Label(self, text="")
                self.wl_bl_label.grid(row=2,column=2,columnspan=2)
                ttk.Button(self, text="Select Solvent Info File", command=self.select_solvent_file).grid(row=3,column=0,columnspan=2)
                self.solvent_label = ttk.Label(self, text="")
                self.solvent_label.grid(row=3, column=2, columnspan=2)
                ttk.Label(self, text="Max VOC ").grid(row=4,column=0)
                self.max_VOC_type = tk.StringVar(self, "wt %")
                ttk.Combobox(self, textvariable = self.max_VOC_type, values=("wt %", "vol %")).grid(row=4,column=3)
                self.max_VOC = ttk.Entry(self)
                self.max_VOC.grid(row=4,column=1,columnspan=2)
                ttk.Label(self, text="# Exempt Solvents: ").grid(row=5,column=0)
                self.solvent_nums = dict()
                self.solvent_nums["min_exempt"] = tk.StringVar(self, "1")
                self.solvent_nums["min_exempt"].trace("w", lambda i, v, o: self.update_solvent_nums("min_exempt"))
                ttk.Combobox(self, textvariable=self.solvent_nums["min_exempt"], values=("1","2")).grid(row=5,column=1)
                ttk.Label(self, text=" to ").grid(row=5,column=2)
                self.solvent_nums["max_exempt"] = tk.StringVar(self, "2")
                self.solvent_nums["max_exempt"].trace("w", lambda i, v, o: self.update_solvent_nums("max_exempt"))
                ttk.Combobox(self, textvariable=self.solvent_nums["max_exempt"], values=("1","2")).grid(row=5,column=3)
                ttk.Label(self, text="# Non-exempt Solvents: ").grid(row=6,column=0)
                self.solvent_nums["min_ne"] = tk.StringVar(self, "1")
                self.solvent_nums["min_ne"].trace("w", lambda i, v, o: self.update_solvent_nums("min_ne"))
                ttk.Combobox(self, textvariable=self.solvent_nums["min_ne"], values=("0","1","2","3")).grid(row=6,column=1)
                ttk.Label(self, text=" to ").grid(row=6,column=2)
                self.solvent_nums["max_ne"] = tk.StringVar(self, "2")
                self.solvent_nums["max_ne"].trace("w", lambda i, v, o: self.update_solvent_nums("max_ne"))
                ttk.Combobox(self, textvariable=self.solvent_nums["max_ne"], values=("0","1","2","3")).grid(row=6,column=3)
                self.target_frame = TargetParamFrame(self)
                self.target_frame.grid(row=7, column=0, columnspan=4)
                ttk.Label(self, text="Formulation Density: ").grid(row=8,column=0)
                self.density_units = tk.StringVar(self, "lb/gal")
                ttk.Combobox(self, textvariable = self.density_units, values=("lb/gal", "g/L")).grid(row=8, column=3)
                self.density_entry = ttk.Entry(self)
                self.density_entry.grid(row=8,column=1,columnspan=2)
                self.temp_frame = TempProfileFrame(self)
                self.temp_frame.grid(row=9, column=0, columnspan=4)
                self.big_style = ttk.Style()
                self.big_style.configure("big.TButton", font=('Arial',14))
                ttk.Label(self, text="Total Time (min): ").grid(row=10,column=0)
                self.total_time = ttk.Entry(self)
                self.total_time.grid(row=10,column=1)
                ttk.Button(self, text="Find Solvent Blends", style="big.TButton", command=self.find_blends).grid(row=11, column=0, columnspan=2)
                self.control_fname = ""
                self.mc_fname = ""
                self.wl_bl_fname = ""
                self.control_blend = None
                self.min_comp = pd.DataFrame({"Name":[],"Volume Fraction":[]})
                self.whitelist = None
                self.using_whitelist = False
                self.blacklist = pd.Series()
                
        def set_main_input(self, main_input):
                self.main_input = main_input
                
        def load_control_blend(self, fname=None):
                if fname is None:
                        fname = tk.filedialog.askopenfilename(filetypes=[("Excel files (.xlsx)", "*.xlsx")])
                self.control_fname = fname
                try:
                        self.control_blend = pd.read_excel(fname)
                except Exception:
                        if fname == "":
                                return
                        tk.messagebox.showwarning(title="Could not Open", message=f"Could not open file {fname}, likely don't have read permission (often happens if the file is in onedrive)")
                        self.focus_force()
                        return
                if "Name" not in self.control_blend.columns or ("Weight Fraction" not in self.control_blend.columns and "Volume Fraction" not in self.control_blend.columns):
                        tk.messagebox.showwarning(title="Invalid Input", message="Incorrectly formatted control blend file")
                        self.control_fname = None
                        self.control_blend = None
                        self.focus_force()
                        return
                self.control_blend_label.configure(text=os.path.basename(fname))
                self.focus_force()
                
        def load_min_comp(self, fname=None):
                if fname is None:
                        fname = tk.filedialog.askopenfilename(filetypes=[("Excel files (.xlsx)", "*.xlsx")])
                try:
                        self.min_comp = pd.read_excel(fname)
                except Exception:
                        if fname == "":
                                return
                        tk.messagebox.showwarning(title="Could not Open", message=f"Could not open file {fname}, likely don't have read permission (often happens if the file is in onedrive)")
                        self.focus_force()
                        return
                self.mc_fname = fname
                if "Name" not in self.min_comp.columns or ("Weight Fraction" not in self.min_comp.columns and "Volume Fraction" not in self.min_comp.columns):
                        tk.messagebox.showwarning(title="Invalid Input", message="Incorrectly formatted minimum composition file")
                        self.mc_fname = None
                        self.min_comp = None
                        self.focus_force()
                        return
                self.min_comp_label.configure(text=os.path.basename(fname))
                self.focus_force()
        def load_wl_bl(self, fname=None):
                if fname is None:
                        fname = tk.filedialog.askopenfilename(filetypes=[("Excel files (.xlsx)", "*.xlsx")])
                try:
                        wl_bl = pd.read_excel(fname)
                except Exception:
                        if fname == "":
                                return
                        tk.messagebox.showwarning(title="Could not Open", message=f"Could not open file {fname}, likely don't have read permission (often happens if the file is in onedrive)")
                        self.focus_force()
                        return
                self.wl_bl_fname = fname
                if len(wl_bl.columns) == 0:
                        tk.messagebox.showwarning(title="Invalid Input", message="Incorrectly formatted whitelist/blacklist file")
                        wl_bl = None
                        self.focus_force()
                        return
                if wl_bl.columns[0].lower() == "whitelist":
                        self.whitelist = wl_bl.iloc[:,0]
                        self.blacklist = None
                        self.using_whitelist = True
                        self.wl_bl_label.configure(text=os.path.basename(fname)+" (Whitelist)")
                elif wl_bl.columns[0].lower() == "blacklist":
                        self.blacklist = wl_bl.iloc[:,0]
                        self.whitelist = None
                        self.using_whitelist = False
                        self.wl_bl_label.configure(text=os.path.basename(fname)+" (Blacklist)")
                else:
                        tk.messagebox.showwarning(title="Invalid Input", message="Incorrectly formatted whitelist/blacklist file")
                        wl_bl = None
                        self.focus_force()
                        return
                self.focus_force()
                
        def update_solvent_nums(self, upd):
                """
                Update the bounds on the min/max number of solvents to use
                """
                updated = self.solvent_nums[upd]
                other = self.solvent_nums["min"+upd[3:] if upd[:3] =="max" else "max"+upd[3:]]
                try:
                        if upd[:3] == "min" and int(updated.get()) > int(other.get()) or upd[:3] == "max" and int(updated.get()) < int(other.get()):
                                other.set(updated.get())
                except ValueError:
                        pass

        def find_blends(self):
                """
                Gathers inputs, checks them, and then calls :func:`get_alternative_blends`
                """
                self.results_frame.clear()
                try:
                        min_exempt = int(self.solvent_nums["min_exempt"].get())
                        max_exempt = int(self.solvent_nums["max_exempt"].get())
                        min_ne = int(self.solvent_nums["min_ne"].get())
                        max_ne = int(self.solvent_nums["max_ne"].get())
                        assert max_exempt > min_exempt
                        assert max_ne > min_ne
                except Exception:
                        tk.messagebox.showwarning(title="Invalid Input", message="Please enter integers (max > min) for min/max solvents to use")
                        self.focus_force()
                        return
                try:
                        max_voc = float(self.max_VOC.get())
                        if self.max_VOC_type.get() in ["vol %", "wt %"]:
                                assert 0 <= max_voc <= 100
                                max_voc /= 100
                except Exception:
                        tk.messagebox.showwarning(title="Invalid Input", message="Enter a decimal number for max VOC (0 < VOC < 100 for wt or vol %)")
                        self.focus_force()
                        return
                target_params = self.target_frame.get_params()
                temp_profile = self.temp_frame.get_temp_profile()
                if target_params is None or temp_profile is None:
                        return
                if self.main_input.all_solvents_df is None:
                        tk.messagebox.showwarning(title="Missing file", message="Missing solvent info file")
                        self.focus_force()
                        return
                if self.using_whitelist:
                        if not all([solvent in self.main_input.all_solvents_df.index for solvent in self.whitelist]):
                                tk.messagebox.showwarning(title="Invalid Input", message="Whitelist incorrectly configured")
                                self.focus_force()
                                return
                elif not all([solvent in self.main_input.all_solvents_df.index for solvent in self.blacklist]):
                        tk.messagebox.showwarning(title="Invalid Input", message="Blacklist incorrectly configured")
                        self.focus_force()
                        return
                replace_by = "Volume Fraction" if "Volume Fraction" in self.min_comp.columns else "Weight Fraction"  
                if not all([solvent in self.main_input.all_solvents_df.index for solvent in self.min_comp["Name"]]) or str(self.min_comp[replace_by].dtype)[:-2] not in ['int','float']:
                        tk.messagebox.showwarning(title="Invalid Input", message="Minimum composition incorrectly configured")
                        self.focus_force()
                        return
                try:
                        formulation_density = float(self.density_entry.get())
                        if self.density_units.get() == "g/L":
                                formulation_density *= 0.0083
                except Exception:
                        tk.messagebox.showwarning(title="Invalid Input", message="Enter a decimal number for formulation density (0 if unknown)")
                        self.focus_force()
                        return

                control_solvents = pd.DataFrame([self.main_input.all_solvents_df.loc[n,:] for n in self.control_blend["Name"]])
                try:
                        if "Volume Fraction" not in self.control_blend.columns:
                                self.control_blend["Volume Fraction"] = self.control_blend["Weight Fraction"] / np.array(control_solvents["Density"])
                                self.control_blend["Volume Fraction"] /= sum(self.control_blend["Volume Fraction"])
                                assert not any(np.isnan(np.array(self.control_blend["Volume Fraction"])))
                        if "Weight Fraction" not in self.control_blend.columns:
                                self.control_blend["Weight Fraction"] = self.control_blend["Volume Fraction"] * np.array(control_solvents["Density"])
                                self.control_blend["Weight Fraction"] /= sum(self.control_blend["Weight Fraction"])
                                assert not any(np.isnan(np.array(self.control_blend["Weight Fraction"])))
                except AssertionError:
                        tk.messagebox.showwarning(title="Invalid Input", message="Volume and/or weight fractions could not be computed - were densities entered correctly?")
                        return
                
                if self.control_blend is None:
                        tk.messagebox.showwarning(title="Missing file", message="Missing control blend file")
                        self.focus_force()
                        return
                if not all([solvent in self.main_input.all_solvents_df.index for solvent in self.control_blend["Name"]]) or str(self.control_blend[replace_by].dtype)[:-2] not in ['int','float']:
                        tk.messagebox.showwarning(title="Invalid Input", message="Control blend incorrectly configured")
                        self.focus_force()
                        return

                VOC_limit_type = self.max_VOC_type.get()
                if VOC_limit_type == "vol %" and replace_by == "Weight Fraction" or VOC_limit_type == "wt %" and replace_by == "Volume Fraction":
                        tk.messagebox.showwarning(title="Invalid Input", message="VOC limit should match minimum composition units")
                        self.focus_force()
                        return
                
                results = get_alternative_blends(self.main_input.all_solvents_df, self.control_blend, self.min_comp, replace_by, target_params,
                                                 temp_profile, (min_exempt, max_exempt), (min_ne, max_ne), (max_voc, VOC_limit_type), formulation_density,
                                                 self.whitelist if self.using_whitelist else None, self.blacklist if not self.using_whitelist else None)
                
                groups = group_similar_results(results, (min_exempt+min_ne, max_exempt+max_ne))
                self.results_frame.make_tree(groups, replace_by)
                
        def select_solvent_file(self):
                self.main_input.select_solvent_file()
                self.focus_force()

        def compare_to_control(self, selected_blend, selected_conc, replace_by):
                """
                Opens the currently selected blend alongside the control blend in the "Compare" window

                Args:
                        selected_conc: Initial concentration of solvents in selected blend (weight %)
                        selected_blend: A list of pandas DataFrame rows containing information about the solvents in the selected blend  
                """
                #Get evaporation data for selected & control blends
                c_comp = [self.main_input.all_solvents_df.loc[name,:] for name in self.control_blend["Name"]]
                c_c0 = self.control_blend["Volume Fraction"]/sum(self.control_blend["Volume Fraction"])
                t_span = [0, float(self.total_time.get())]
                target = self.target_frame.get_params()
                temp_curve = self.temp_frame.get_temp_profile()
                c_t, c_total_profile, c_partial_profiles, c_RED = get_evap_curve(c_c0, c_comp, target, temp_curve, t_span, self.main_input.all_solvents_df, replace_by == "Weight Fraction")
                a_comp = [self.main_input.all_solvents_df.loc[name,:] for name in selected_blend] + [self.main_input.all_solvents_df.loc[name,:] for name in self.min_comp["Name"]]
                if replace_by == "Weight Fraction":
                        a_wf = np.array(list(selected_conc)+list(self.min_comp["Weight Fraction"]))
                        a_c0 = a_wf * np.array(pd.DataFrame(a_comp)["Density"])
                        a_c0 /= sum(a_c0)
                else:
                        a_c0 = np.array(list(selected_conc)+list(self.min_comp["Volume Fraction"]))
                a_t, a_total_profile, a_partial_profiles, a_RED = get_evap_curve(a_c0, a_comp, target, temp_curve, t_span, self.main_input.all_solvents_df, replace_by == "Weight Fraction")
                
                #Make a folder in ./tmp to store Excel outputs 
                tmp_dir = f"./tmp/{str(uuid.uuid4())}"
                os.mkdir(tmp_dir)

                #Write data to files in folder
                control_path = f"{tmp_dir}/Control.xlsx"
                write_to_excel(control_path, self.control_blend["Name"], c_t, c_total_profile, c_partial_profiles, c_RED, temp_curve(c_t), caption=replace_by)
                alternative_path = f"{tmp_dir}/Alternative ({', '.join(selected_blend)}).xlsx"
                write_to_excel(alternative_path, selected_blend+list(self.min_comp["Name"]), a_t, a_total_profile, a_partial_profiles, a_RED, temp_curve(a_t), caption=replace_by)

                #Open compare screen with files
                self.compare_input.clear()
                self.compare_input.add_files([control_path, alternative_path])
                self.compare_screen.deiconify()
                
        def export_output(self, selected_blend, selected_conc, replace_by):
                """
                Exports the currently selected blend to an Excel sheet

                Args:
                        selected_conc: Initial concentration of solvents in selected blend (weight %)
                        selected_blend: A list of pandas DataFrame rows containing information about the solvents in the selected blend  
                """
                target = self.target_frame.get_params()
                temp_curve = self.temp_frame.get_temp_profile()
                t_span = [0, float(self.total_time.get())]
                a_comp = [self.main_input.all_solvents_df.loc[name,:] for name in selected_blend] + [self.main_input.all_solvents_df.loc[name,:] for name in self.min_comp["Name"]]
                if replace_by == "Weight Fraction":
                        a_wf = np.array(list(selected_conc)+list(self.min_comp["Weight Fraction"]))
                        a_c0 = a_wf * np.array(pd.DataFrame(a_comp)["Density"])
                        a_c0 /= sum(a_c0)
                else:
                        a_c0 = np.array(list(selected_conc)+list(self.min_comp["Volume Fraction"]))
                a_t, a_total_profile, a_partial_profiles, a_RED = get_evap_curve(a_c0, a_comp, target, temp_curve, t_span, self.main_input.all_solvents_df, replace_by == "Weight Fraction")
                alternative_path = tk.filedialog.asksaveasfilename()
                self.focus_force()
                write_to_excel(alternative_path, selected_blend+list(self.min_comp["Name"]), a_t, a_total_profile, a_partial_profiles, a_RED, temp_curve(a_t), caption=replace_by)
                if alternative_path[-5:] != ".xlsx":
                        alternative_path = alternative_path + ".xlsx"
                os.startfile(alternative_path)
        def get_config(self):
                return {"control_fname": self.control_fname,
                        "mc_fname": self.mc_fname,
                        "wl_bl_fname": self.wl_bl_fname,
                        "temp_frame": self.temp_frame.get_config(),
                        "target_frame": self.target_frame.get_config(),
                        "total_time": self.total_time.get(),
                        "voc_max": self.max_VOC.get()}
                        
        def set_config(self, config):
                if config["control_fname"] != "":
                        self.load_control_blend(config["control_fname"])
                if config["mc_fname"] != "":
                        self.load_min_comp(config["mc_fname"])
                if config["wl_bl_fname"] != "":
                        self.load_wl_bl(config["wl_bl_fname"])
                self.target_frame.set_config(config["target_frame"])
                self.temp_frame.set_config(config["temp_frame"])
                self.total_time.delete(0, tk.END)
                self.total_time.insert(0, config["total_time"])
                self.max_VOC.delete(0, tk.END)
                self.max_VOC.insert(0, config["voc_max"])
                
class ReformResultsFrame(ttk.Frame):
        """
        Displays results (found alternative blends) and facilitates exporting data/comparison to control for them.
        """
        def __init__(self, root):
                super().__init__(root)
                self.big_style = ttk.Style()
                self.big_style.configure("big.TLabel", font=('Arial',14))
                self.big_style.configure("big.TButton", font=('Arial',14))
                ttk.Label(self, text="Alternative Blends", style="big.TLabel").grid(row=0,column=0)
                self.display = ttk.Treeview(self, show='tree')
                self.display.column('#0', width=500)
                self.display.grid(row=1, column=0)
                self.display.bind("<<TreeviewSelect>>", self.update_selection)
                self.export_button = ttk.Button(self, text="Export to Excel", state="disabled", style="big.TButton", command=self.export_output)
                self.export_button.grid(row=2, column=0)
                self.compare_button = ttk.Button(self, text="Compare to Control", state="disabled", style="big.TButton", command=self.compare_to_control)
                self.compare_button.grid(row=3,column=0)

        def set_input_frame(self, input_frame):
                self.input_frame = input_frame
        
        def compare_to_control(self):
                self.input_frame.compare_to_control(self.selected_blend, self.selected_conc, self.replace_by)

        def export_output(self):
                self.input_frame.export_output(self.selected_blend, self.selected_conc, self.replace_by)
                
        def update_selection(self, _):
                """
                Updates which blend is currently selected.
                Behavior:

                - If a "results" item or one of its children is selected, use that solvent blend
                - If the currently selected solvent has results, use the blend up to that point
                - Otherwise, step down the tree (using the first child of each element) until we reach a blend with results. This will return the lowest cost blend that contains our selection.
                """
                self.compare_button.configure(state="normal")
                self.export_button.configure(state="normal")
                cur_id = self.display.selection()[0]
                cur_item = self.display.item(cur_id)
                names = []
                while self.display.parent(cur_id) != "":
                        names.append(cur_item["text"])
                        cur_id = self.display.parent(cur_id)
                        cur_item = self.display.item(cur_id)
                names.append(cur_item["text"])
                cur_reference = self.grouped_results
                blend = []
                for name in names[::-1]:
                        if name == "Results":
                                self.selected_blend = blend
                                self.selected_conc = cur_reference["result"]["conc"]
                                return
                        blend.append(name)
                        cur_reference = cur_reference[name]
                while "result" not in cur_reference.keys():
                        next_key = list(cur_reference.keys())[0]
                        blend.append(next_key)
                        cur_reference = cur_reference[next_key]
                self.selected_blend = blend
                self.selected_conc = cur_reference["result"]["conc"]
        def make_tree(self, grouped_results, replace_by):
                self.grouped_results = grouped_results
                self.replace_by = replace_by
                cur_iid = 0
                cur_items = self.grouped_results.items()
                while len(cur_items) > 0:
                        new_cur = []
                        for key, value in cur_items:
                                if key == "result":
                                        self.display.insert(value.get("parent"), tk.END, iid=cur_iid, text="Results")
                                        self.display.insert(cur_iid, tk.END, cur_iid + 1, text=f"Cost: {value['cost']}")
                                        self.display.insert(cur_iid, tk.END, cur_iid + 2, text=f"{self.replace_by}s: {', '.join([str(x) for x in list(np.round(value['conc']/sum(value['conc']), 2))])}") 
                                        cur_iid += 3
                                else:
                                        self.display.insert('' if value.get("parent") is None else value.get("parent"), tk.END, iid=cur_iid, text=key)
                                        for subtree in value.items():
                                                if subtree[0] != "parent":
                                                        subtree[1]["parent"] = cur_iid
                                                        new_cur.append(subtree)
                                        cur_iid += 1
                        cur_items = new_cur
        def clear(self):
                self.display.destroy()
                self.selected_blend = None
                self.selected_conc = None
                self.compare_button.configure(state="disabled")
                self.export_button.configure(state="disabled")
                self.grouped_results = None
                self.display = ttk.Treeview(self, show='tree')
                self.display.column('#0', width=500)
                self.display.grid(row=1, column=0)
                self.display.bind("<<TreeviewSelect>>", self.update_selection)
        
                
                
class GraphFrame(ttk.Frame):
        """
        3 primary graphs: concentration (wt%), RED, and temperature profile. Used on main & comparison screens
        """
        def __init__(self, root):
                super().__init__(root)
                self.fig = Figure(figsize=(5, 6), dpi=100)
                self.fig.set_layout_engine("tight")
                self.conc_ax = self.fig.add_subplot(5,1,(1,3))
                self.conc_ax.set_xlabel("time [min]")
                self.conc_ax.set_ylabel("wt frac")
                self.conc_ax.set_title("Solvent Concentration")
                self.RED_ax = self.fig.add_subplot(5,1,4)
                self.RED_ax.set_xlabel("time [min]")
                self.RED_ax.set_ylabel("RED")
                self.RED_ax.set_title("RED")
                self.temp_ax = self.fig.add_subplot(5,1,5)
                self.temp_ax.set_xlabel("time [min]")
                self.temp_ax.set_ylabel("Temperature")
                self.temp_ax.set_title("Temperature Profile")
                self.canvas = FigureCanvasTkAgg(self.fig, master=self)
                self.canvas.draw()
                self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
                self.toolbar.update()
                self.canvas.mpl_connect("key_press_event", key_press_handler)
                self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        def clear_artists(self):
                for artist in self.temp_ax.lines + self.temp_ax.collections:
                    artist.remove()
                for artist in self.RED_ax.lines + self.RED_ax.collections:
                    artist.remove()
                for artist in self.conc_ax.lines + self.conc_ax.collections:
                    artist.remove()

if __name__ == "__main__":
        root = tk.Tk()
        root.wm_title("Extended Evaporation Simulator")
        compare_screen = tk.Toplevel(root)
        compare_screen.wm_title("Evaporation Simulator Comparison Tool")
        compare_graph = GraphFrame(compare_screen)
        compare_input = CompareInputFrame(compare_screen, compare_graph)
        compare_input.grid(row=0, column=0, sticky="N")
        compare_graph.grid(row=0, column=1)
        compare_screen.withdraw()
        compare_screen.protocol("WM_DELETE_WINDOW", compare_screen.withdraw)
        reform_screen = tk.Toplevel(root)
        reform_results = ReformResultsFrame(reform_screen)
        reform_results.grid(row=0,column=1)
        reform_input = ReformInputFrame(reform_screen, compare_screen, compare_input, reform_results)
        reform_input.grid(row=0,column=0)
        reform_results.set_input_frame(reform_input)
        reform_screen.withdraw()
        reform_screen.protocol("WM_DELETE_WINDOW", reform_screen.withdraw)
        
        graph_frame = GraphFrame(root)
        input_frame = MainInputFrame(root, graph_frame, compare_input, compare_screen, reform_input, reform_screen)
        input_frame.grid(column=0,row=0, sticky="N")
        graph_frame.grid(column=1,row=0)
        root.mainloop()