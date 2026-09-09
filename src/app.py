import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from video_processor import VideoProcessor


ALLOWED_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


class BirdWatcherApp:
	def __init__(self, root: tk.Tk):
		self.root = root
		self.root.title("BirdWatcher")
		self.root.minsize(560, 360)
		self.root.protocol("WM_DELETE_WINDOW", self.close)

		self.video_path = tk.StringVar()
		self.output_dir = tk.StringVar()
		self.sample_interval = tk.StringVar(value="1.0")
		self.use_gpu = tk.BooleanVar(value=False)
		self.batch_size = tk.StringVar(value="1")
		self.box_threshold = tk.StringVar(value="50")
		self.class_threshold = tk.StringVar(value="50")
		self.detection_progress = tk.DoubleVar(value=0)
		self.classification_progress = tk.DoubleVar(value=0)
		self.output_progress = tk.DoubleVar(value=0)
		self.status = tk.StringVar(value="Ready")
		self.advanced_visible = False
		self.processing = False
		self.worker = None
		self.cancel_event = None
		self.messages = queue.Queue()

		self._build_ui()
		self.stage_labels = {
			"Finding birds": self.detection_stage_label,
			"Classifying birds": self.classification_stage_label,
			"Saving output": self.output_stage_label,
		}
		self._set_active_stage(None)
		self.root.after(100, self._poll_messages)

	def _build_ui(self):
		container = ttk.Frame(self.root, padding=16)
		container.pack(fill="both", expand=True)
		container.columnconfigure(1, weight=1)

		ttk.Label(container, text="Video").grid(row=0, column=0, sticky="w", pady=6)
		ttk.Entry(container, textvariable=self.video_path).grid(
			row=0, column=1, sticky="ew", padx=(12, 8), pady=6
		)
		self.video_button = ttk.Button(
			container, text="Browse...", command=self._choose_video
		)
		self.video_button.grid(row=0, column=2, pady=6)
		self._info_button(container, 0, "Video file", "Select the video to process.")

		ttk.Label(container, text="Output directory").grid(
			row=1, column=0, sticky="w", pady=6
		)
		ttk.Entry(container, textvariable=self.output_dir).grid(
			row=1, column=1, sticky="ew", padx=(12, 8), pady=6
		)
		self.output_button = ttk.Button(
			container, text="Browse...", command=self._choose_output
		)
		self.output_button.grid(row=1, column=2, pady=6)
		self._info_button(
			container, 1, "Output directory", "Select where processed images will be saved."
		)

		ttk.Label(container, text="Sampling interval (seconds)").grid(
			row=2, column=0, sticky="w", pady=6
		)
		ttk.Entry(container, textvariable=self.sample_interval, width=12).grid(
			row=2, column=1, sticky="w", padx=(12, 8), pady=6
		)
		self._info_button(
			container, 2, "Sampling interval", "Template description for the sampling interval."
		)

		self.advanced_button = ttk.Button(
			container, text="Show advanced settings", command=self._toggle_advanced
		)
		self.advanced_button.grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 4))

		self.advanced_frame = ttk.Frame(container)
		self.advanced_frame.grid(row=4, column=0, columnspan=4, sticky="ew")
		self.advanced_frame.columnconfigure(1, weight=1)
		self._build_advanced_settings()
		self.advanced_frame.grid_remove()

		ttk.Separator(container).grid(
			row=5, column=0, columnspan=4, sticky="ew", pady=(18, 12)
		)
		self.detection_stage_label = tk.Label(container, text="Finding birds")
		self.detection_stage_label.grid(
			row=6, column=0, columnspan=4, sticky="w", pady=(0, 4)
		)
		ttk.Progressbar(
			container, variable=self.detection_progress, maximum=100
		).grid(row=7, column=0, columnspan=4, sticky="ew", pady=(0, 10))
		self.classification_stage_label = tk.Label(container, text="Classifying birds")
		self.classification_stage_label.grid(
			row=8, column=0, columnspan=4, sticky="w", pady=(0, 4)
		)
		ttk.Progressbar(
			container, variable=self.classification_progress, maximum=100
		).grid(row=9, column=0, columnspan=4, sticky="ew", pady=(0, 8))
		self.output_stage_label = tk.Label(container, text="Saving output")
		self.output_stage_label.grid(
			row=10, column=0, columnspan=4, sticky="w", pady=(0, 4)
		)
		ttk.Progressbar(
			container, variable=self.output_progress, maximum=100
		).grid(row=11, column=0, columnspan=4, sticky="ew", pady=(0, 8))
		tk.Label(container, textvariable=self.status).grid(
			row=12, column=0, columnspan=3, sticky="w"
		)
		self.process_button = ttk.Button(
			container, text="Process", command=self._start_or_cancel
		)
		self.process_button.grid(row=13, column=0, columnspan=4, pady=(18, 0))

	def _build_advanced_settings(self):
		ttk.Checkbutton(self.advanced_frame, text="Use GPU", variable=self.use_gpu).grid(
			row=0, column=0, columnspan=2, sticky="w", pady=5
		)
		self._info_button(
			self.advanced_frame, 0, "Use GPU", "Template description for GPU acceleration.", column=2
		)

		ttk.Label(self.advanced_frame, text="Batch size").grid(
			row=1, column=0, sticky="w", pady=5
		)
		self.batch_menu = ttk.Combobox(
			self.advanced_frame,
			textvariable=self.batch_size,
			values=[str(size) for size in ALLOWED_BATCH_SIZES],
			state="readonly",
			width=10,
		)
		self.batch_menu.grid(row=1, column=1, sticky="w", padx=(12, 8), pady=5)
		self._info_button(
			self.advanced_frame, 1, "Batch size", "Template description for the batch size.", column=2
		)

		self._add_threshold_setting(
			row=2,
			label="Box confidence threshold",
			variable=self.box_threshold,
			title="Box confidence threshold",
		)
		self._add_threshold_setting(
			row=3,
			label="Class confidence threshold",
			variable=self.class_threshold,
			title="Class confidence threshold",
		)

	def _add_threshold_setting(self, row, label, variable, title):
		ttk.Label(self.advanced_frame, text=label).grid(
			row=row, column=0, sticky="w", pady=5
		)
		ttk.Scale(
			self.advanced_frame,
			from_=0,
			to=100,
			variable=variable,
			orient="horizontal",
		).grid(row=row, column=1, sticky="ew", padx=(12, 8), pady=5)
		ttk.Entry(self.advanced_frame, textvariable=variable, width=6).grid(
			row=row, column=2, sticky="e", pady=5
		)
		self._info_button(
			self.advanced_frame,
			row,
			title,
			f"Template description for {label.lower()}.",
			column=3,
		)

	@staticmethod
	def _info_button(parent, row, title, description, column=3):
		ttk.Button(
			parent,
			text="ⓘ",
			width=3,
			command=lambda: messagebox.showinfo(title, description),
		).grid(row=row, column=column, padx=(4, 0), pady=5)

	def _choose_video(self):
		selected = filedialog.askopenfilename(
			title="Select video",
			filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
		)
		if selected:
			self.video_path.set(selected)

	def _choose_output(self):
		selected = filedialog.askdirectory(title="Select output directory")
		if selected:
			self.output_dir.set(selected)

	def _toggle_advanced(self):
		self.advanced_visible = not self.advanced_visible
		if self.advanced_visible:
			self.advanced_frame.grid()
			self.advanced_button.configure(text="Hide advanced settings")
		else:
			self.advanced_frame.grid_remove()
			self.advanced_button.configure(text="Show advanced settings")

	def _start_or_cancel(self):
		if self.processing:
			self.cancel_event.set()
			self.status.set("Cancelling...")
			self.process_button.configure(state="disabled")
			return
		self._start_processing()

	def _start_processing(self):
		values = self._read_and_validate()
		if values is None:
			return

		(
			video_path,
			output_dir,
			sample_interval,
			batch_size,
			box_threshold,
			class_threshold,
		) = values
		try:
			output_dir.mkdir(parents=True, exist_ok=True)
		except OSError as error:
			messagebox.showerror("Invalid output directory", str(error))
			return
		use_gpu = self.use_gpu.get()
		self.cancel_event = threading.Event()
		self.processing = True
		self.detection_progress.set(0)
		self.classification_progress.set(0)
		self.output_progress.set(0)
		self._set_active_stage("Finding birds")
		self.status.set("")
		self.process_button.configure(text="Cancel")
		self._set_controls_enabled(False)
		self.process_button.configure(state="normal")
		self.worker = threading.Thread(
			target=self._process_video,
			args=(
				video_path,
				output_dir,
				sample_interval,
				use_gpu,
				batch_size,
				box_threshold / 100,
				class_threshold / 100,
			),
			daemon=True,
		)
		self.worker.start()

	def _read_and_validate(self):
		video_text = self.video_path.get().strip()
		output_text = self.output_dir.get().strip()
		if not output_text:
			messagebox.showerror("Invalid output directory", "Select an output directory.")
			return None
		video = Path(video_text)
		output = Path(output_text)
		if not video.is_file():
			messagebox.showerror("Invalid video", "Select an existing video file.")
			return None
		try:
			interval = float(self.sample_interval.get())
		except ValueError:
			messagebox.showerror("Invalid sampling interval", "Enter a positive number of seconds.")
			return None
		if interval <= 0:
			messagebox.showerror("Invalid sampling interval", "Enter a positive number of seconds.")
			return None
		try:
			batch_size = int(self.batch_size.get())
		except ValueError:
			messagebox.showerror("Invalid batch size", "Select a valid batch size.")
			return None
		if batch_size not in ALLOWED_BATCH_SIZES:
			messagebox.showerror("Invalid batch size", "Select a valid batch size.")
			return None
		try:
			box_threshold = float(self.box_threshold.get())
			class_threshold = float(self.class_threshold.get())
		except ValueError:
			messagebox.showerror("Invalid confidence threshold", "Enter a value from 0 to 100.")
			return None
		if not 0 <= box_threshold <= 100 or not 0 <= class_threshold <= 100:
			messagebox.showerror("Invalid confidence threshold", "Enter a value from 0 to 100.")
			return None
		return video, output, interval, batch_size, box_threshold, class_threshold

	def _process_video(
		self,
		video_path,
		output_dir,
		sample_interval,
		use_gpu,
		batch_size,
		box_threshold,
		class_threshold,
	):
		try:
			result = VideoProcessor().process(
				video_path=video_path,
				output_dir=output_dir,
				sample_interval=sample_interval,
				use_gpu=use_gpu,
				batch_size=batch_size,
				box_conf_threshold=box_threshold,
				class_conf_threshold=class_threshold,
				progress_callback=lambda fraction, stage: self.messages.put(
					("progress", fraction, stage)
				),
				cancel_event=self.cancel_event,
			)
			self.messages.put(("cancelled" if result is False else "complete",))
		except Exception as error:
			self.messages.put(("error", str(error)))

	def _poll_messages(self):
		try:
			while True:
				message = self.messages.get_nowait()
				kind = message[0]
				if kind == "progress":
					_, fraction, stage = message
					progress = max(0, min(100, fraction * 100))
					self._set_active_stage(stage)
					if stage == "Finding birds":
						self.detection_progress.set(progress)
					elif stage == "Classifying birds":
						self.detection_progress.set(100)
						self.classification_progress.set(progress)
					else:
						self.classification_progress.set(100)
						self.output_progress.set(progress)
				elif kind == "complete":
					self._finish("Complete", 100)
				elif kind == "cancelled":
					self._finish("Cancelled", None)
				elif kind == "error":
					self._finish("Processing failed", None)
					messagebox.showerror("Processing failed", message[1])
		except queue.Empty:
			pass
		self.root.after(100, self._poll_messages)

	def _finish(self, status, progress):
		self.processing = False
		self.worker = None
		if progress is not None:
			self.detection_progress.set(progress)
			self.classification_progress.set(progress)
			self.output_progress.set(progress)
		else:
			self.detection_progress.set(0)
			self.classification_progress.set(0)
			self.output_progress.set(0)
		self._set_active_stage(None)
		self.status.set(status)
		self.process_button.configure(text="Process")
		self._set_controls_enabled(True)

	def _set_active_stage(self, active_stage):
		for stage, label in self.stage_labels.items():
			label.configure(fg="black" if active_stage in (None, stage) else "gray")

	def _set_controls_enabled(self, enabled):
		state = "normal" if enabled else "disabled"
		for widget in (self.video_button, self.output_button, self.advanced_button):
			widget.configure(state=state)
		self._set_children_state(self.root, state)
		self.process_button.configure(state="normal" if not enabled else state)

	@staticmethod
	def _set_children_state(widget, state):
		for child in widget.winfo_children():
			if child.winfo_class() not in {"TProgressbar", "TSeparator", "TLabel"}:
				try:
					child.configure(state=state)
				except tk.TclError:
					pass
			BirdWatcherApp._set_children_state(child, state)

	def close(self):
		if self.processing:
			self.cancel_event.set()
		self.root.destroy()


def main():
	root = tk.Tk()
	BirdWatcherApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
