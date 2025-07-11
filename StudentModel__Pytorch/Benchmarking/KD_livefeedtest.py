#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==============================================================================
#                      CELL 1: SETUP & GPU CHECK (Corrected)
# ==============================================================================
# This cell imports libraries and verifies that a GPU is active.

import os
import io
from collections import OrderedDict

import torch
import torch.nn as nn
from google.colab import files
from torchvision import transforms  # <-- THIS IS THE MISSING LINE TO ADD

# --- Check for GPU ---
# This line automatically selects the GPU if available, otherwise falls back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  print(f"✅ GPU is active and ready! Device: {torch.cuda.get_device_name(0)}")
else:
  print("⚠️ WARNING: No GPU detected. The live demo will be very slow.")
  print("Go to 'Runtime' -> 'Change runtime type' and select 'GPU' as the hardware accelerator.")


# In[ ]:


# ==============================================================================
#               CELL 2: STUDENT MODEL ARCHITECTURE DEFINITION
# ==============================================================================
# This cell contains the Python classes that define the RRDBNet model structure.

DOWNSCALE_FACTOR = 4

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=32, gc=16):
        super(ResidualDenseBlock,self).__init__();self.conv1=nn.Conv2d(nf,gc,3,1,1);self.conv2=nn.Conv2d(nf+gc,gc,3,1,1);self.conv3=nn.Conv2d(nf+2*gc,gc,3,1,1);self.conv4=nn.Conv2d(nf+3*gc,gc,3,1,1);self.conv5=nn.Conv2d(nf+4*gc,nf,3,1,1);self.lrelu=nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self,x):
        x1=self.lrelu(self.conv1(x));x2=self.lrelu(self.conv2(torch.cat((x,x1),1)));x3=self.lrelu(self.conv3(torch.cat((x,x1,x2),1)));x4=self.lrelu(self.conv4(torch.cat((x,x1,x2,x3),1)));x5=self.conv5(torch.cat((x,x1,x2,x3,x4),1));return x5*0.2+x

class RRDB(nn.Module):
    def __init__(self,nf,gc=16):
        super(RRDB,self).__init__();self.RDB1=ResidualDenseBlock(nf,gc);self.RDB2=ResidualDenseBlock(nf,gc);self.RDB3=ResidualDenseBlock(nf,gc)
    def forward(self,x):
        out=self.RDB1(x);out=self.RDB2(out);out=self.RDB3(out);return out*0.2+x

class RRDBNet_v9(nn.Module):
    def __init__(self,in_nc=3,out_nc=3,nf=32,nb=4,gc=16,upscale=4):
        super(RRDBNet_v9,self).__init__();self.conv_first=nn.Conv2d(in_nc,nf,3,1,1);self.body=nn.Sequential(*[RRDB(nf,gc=gc) for _ in range(nb)]);self.conv_body=nn.Conv2d(nf,nf,3,1,1);self.upsampler=nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(nf,nf,3,1,1),nn.LeakyReLU(0.2,inplace=True),nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(nf,nf,3,1,1),nn.LeakyReLU(0.2,inplace=True));self.conv_hr=nn.Conv2d(nf,nf,3,1,1);self.conv_last=nn.Conv2d(nf,out_nc,3,1,1);self.lrelu=nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self,x):
        fea=self.conv_first(x);trunk=self.conv_body(self.body(fea));fea=fea+trunk;fea=self.upsampler(fea);fea=self.lrelu(self.conv_hr(fea));out=self.conv_last(fea);return torch.sigmoid(out)

print("✅ Student model architecture (RRDBNet_v9) defined.")


# In[ ]:


# ==============================================================================
#               CELL 3: LOAD THE TRAINED STUDENT MODEL
# ==============================================================================
# This cell instantiates the model and loads your trained .pth weights file.

# Create an empty instance of the model and send it to the GPU
model = RRDBNet_v9(nf=32, nb=4, gc=16, upscale=DOWNSCALE_FACTOR).to(DEVICE)

print("Please upload your trained student model (.pth file).")
uploaded = files.upload()

if not uploaded:
    print("❌ Upload cancelled. Please run the cell again to upload the model.")
else:
    try:
        model_filename = list(uploaded.keys())[0]
        model_data = uploaded[model_filename]

        # Load the state dictionary from the uploaded file
        state_dict = torch.load(io.BytesIO(model_data), map_location=DEVICE)

        # This handles models that were trained with DataParallel (adds 'module.' prefix)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        # Set the model to evaluation mode. This is crucial for consistent results.
        model.eval()

        print(f"✅ Model '{model_filename}' loaded successfully and is ready for the demo.")

    except Exception as e:
        print(f"❌ FATAL ERROR loading model: {e}")
        print("Please ensure you uploaded the correct .pth file and that the architecture matches.")


# In[ ]:


# ==============================================================================
#      FINAL LIVE SIDE-BY-SIDE DEMO (Robust Single-Cell Version)
# ==============================================================================
# This single cell handles webcam initialization and the real-time enhancement
# loop, displaying the original and enhanced feeds side-by-side.

# --- Imports for this specific demo ---
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
import cv2
import numpy as np
import time
from torchvision import transforms
import io

# We assume 'model', 'DEVICE', and other necessary variables exist from setup.
# ------------------------------------------------------------------------------

def pil_to_b64(pil_img):
    """Converts a PIL Image to a base64 encoded string for HTML display."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return b64encode(buffered.getvalue()).decode('utf-8')

def live_side_by_side_demo():
    """
    Main function to run the entire live demo with side-by-side comparison.
    """
    # --- 1. Define all necessary JavaScript and HTML ---
    js_code = """
    var video;
    var stream;

    // Function to set up the HTML layout and start the camera
    async function setupLayoutAndCamera() {
      // Create hidden video element for capturing
      video = document.createElement('video');
      video.id = 'camera-video';
      video.setAttribute('playsinline', '');
      video.style.display = 'none';

      // Attach the video to the document
      const div = document.createElement('div');
      div.appendChild(video);
      document.body.appendChild(div);

      // Start the stream
      stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream;
      await video.play();

      return "Camera is ready!";
    }

    // Function to capture a single frame from the running stream
    async function captureFrame() {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.8);
    }

    // Function to update the side-by-side image elements
    function updateImageViews(original_b64, enhanced_b64) {
      document.getElementById('original-view').src = 'data:image/jpeg;base64,' + original_b64;
      document.getElementById('enhanced-view').src = 'data:image/jpeg;base64,' + enhanced_b64;
    }

    // Function to stop the camera stream
    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    }
    """
    # --- Define the HTML structure for the side-by-side display ---
    html_layout = """
    <div style="display: flex; justify-content: space-around;">
      <div style="text-align: center;">
        <h3>Original Feed</h3>
        <img id="original-view" width="480">
      </div>
      <div style="text-align: center;">
        <h3>Enhanced Feed (w/ FPS)</h3>
        <img id="enhanced-view" width="480">
      </div>
    </div>
    """

    # --- 2. Inject JS/HTML and start the camera ---
    display(HTML(html_layout))
    display(Javascript(js_code))
    print("Initializing webcam... Please allow camera access.")
    try:
        eval_js('setupLayoutAndCamera()')
        print("✅ Webcam is active. Starting enhancement loop...")
    except Exception as e:
        print(f"❌ Could not start camera. Error: {e}")
        return

    # --- 3. Run the main enhancement loop ---
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    MODEL_INPUT_SIZE = (480, 270) # Width, Height

    try:
        while True:
            # A. Capture a raw frame and convert it for display
            b64_data_raw = eval_js('captureFrame()')
            original_pil = PIL.Image.open(io.BytesIO(b64decode(b64_data_raw.split(',')[1])))
            original_b64_display = pil_to_b64(original_pil) # For the left-side view

            # B. Prepare the frame for the model
            input_for_model_pil = original_pil.resize(MODEL_INPUT_SIZE, PIL.Image.BICUBIC)
            input_tensor = to_tensor(input_for_model_pil).unsqueeze(0).to(DEVICE)

            # C. Timed Inference
            start_time = time.time()
            with torch.no_grad():
                output_tensor = model(input_tensor)
            proc_time = time.time() - start_time
            fps = 1.0 / proc_time

            # D. Post-process, draw FPS text, and convert for display
            # THE FIX IS HERE: The indentation is now correct.
            enhanced_pil = to_pil(output_tensor.clamp(0, 1).squeeze(0).cpu())
            enhanced_cv = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
            cv2.putText(
                enhanced_cv,
                f"Model FPS: {fps:.2f}",
                (20, 60), # Position adjusted for larger text
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3 # Larger font
            )
            _, buffer = cv2.imencode('.jpg', enhanced_cv)
            enhanced_b64_display = b64encode(buffer).decode('utf-8')

            # E. Call JS to update both images at once
            eval_js(f"updateImageViews('{original_b64_display}', '{enhanced_b64_display}')")

    except KeyboardInterrupt:
        print("\nLoop stopped by user.")
    except Exception as e:
        print(f"Loop stopped due to an error: {e}")
    finally:
        # F. Clean up and stop the camera
        print("Attempting to stop webcam stream...")
        eval_js('stopCamera()')
        print("Webcam stream has been stopped.")

# --- Run the main demo function ---
live_side_by_side_demo()


# In[ ]:


# ==============================================================================
#      FINAL LIVE SIDE-BY-SIDE DEMO (Dual FPS Measurement Version)
# ==============================================================================
# This version calculates and displays both the Model's pure speed and the
# system's real-time end-to-end throughput.

# --- Imports for this specific demo ---
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
import cv2
import numpy as np
import time
from torchvision import transforms
import io

# We assume 'model', 'DEVICE', and other necessary variables exist from setup.
# ------------------------------------------------------------------------------

def pil_to_b64(pil_img):
    """Converts a PIL Image to a base64 encoded string for HTML display."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return b64encode(buffered.getvalue()).decode('utf-8')

def live_side_by_side_demo():
    """
    Main function to run the entire live demo with side-by-side comparison.
    """
    # --- 1. Define all necessary JavaScript and HTML ---
    js_code = """
    var video;
    var stream;
    async function setupLayoutAndCamera() { /* ... JS code is the same ... */ }
    async function captureFrame() { /* ... JS code is the same ... */ }
    function updateImageViews(original_b64, enhanced_b64) { /* ... JS code is the same ... */ }
    function stopCamera() { /* ... JS code is the same ... */ }
    """
    # For brevity, I've collapsed the JS code. Use the full block from the previous answer.
    # The full, correct JS code from the previous answer is used here.
    js_code = """
    var video;
    var stream;
    async function setupLayoutAndCamera() { video = document.createElement('video'); video.id = 'camera-video'; video.setAttribute('playsinline', ''); video.style.display = 'none'; const div = document.createElement('div'); div.appendChild(video); document.body.appendChild(div); stream = await navigator.mediaDevices.getUserMedia({video: true}); video.srcObject = stream; await video.play(); return "Camera is ready!"; }
    async function captureFrame() { const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight; canvas.getContext('2d').drawImage(video, 0, 0); return canvas.toDataURL('image/jpeg', 0.8); }
    function updateImageViews(original_b64, enhanced_b64) { document.getElementById('original-view').src = 'data:image/jpeg;base64,' + original_b64; document.getElementById('enhanced-view').src = 'data:image/jpeg;base64,' + enhanced_b64; }
    function stopCamera() { if (stream) { stream.getTracks().forEach(track => track.stop()); } }
    """

    html_layout = """
    <div style="display: flex; justify-content: space-around;">
      <div style="text-align: center;">
        <h3>Original Feed</h3>
        <img id="original-view" width="480">
      </div>
      <div style="text-align: center;">
        <h3>Enhanced Feed (w/ FPS)</h3>
        <img id="enhanced-view" width="480">
      </div>
    </div>
    """

    # --- 2. Inject JS/HTML and start the camera ---
    display(HTML(html_layout))
    display(Javascript(js_code))
    print("Initializing webcam... Please allow camera access.")
    try:
        eval_js('setupLayoutAndCamera()')
        print("✅ Webcam is active. Starting enhancement loop...")
    except Exception as e:
        print(f"❌ Could not start camera. Error: {e}")
        return

    # --- 3. Run the main enhancement loop ---
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    MODEL_INPUT_SIZE = (480, 270)  # Width, Height

    try:
        while True:
            # === Start full loop timer ===
            loop_start_time = time.time()

            # === Capture and Prepare ===
            b64_data_raw = eval_js('captureFrame()')
            original_pil = PIL.Image.open(io.BytesIO(b64decode(b64_data_raw.split(',')[1])))
            original_b64_display = pil_to_b64(original_pil)
            input_for_model_pil = original_pil.resize(MODEL_INPUT_SIZE, PIL.Image.BICUBIC)
            input_tensor = to_tensor(input_for_model_pil).unsqueeze(0).to(DEVICE)

            # === Timed Model Inference ===
            inference_start_time = time.time()
            with torch.no_grad():
                output_tensor = model(input_tensor)
            inference_duration = time.time() - inference_start_time
            model_fps = 1.0 / inference_duration

            # === Post-process ===
            enhanced_pil = to_pil(output_tensor.clamp(0, 1).squeeze(0).cpu())
            enhanced_cv = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)

            # === Stop full loop timer and calculate Loop FPS ===
            loop_duration = time.time() - loop_start_time
            loop_fps = 1.0 / loop_duration

            # === Overlay BOTH FPS text values ===
            cv2.putText(
                enhanced_cv,
                f"Model FPS: {model_fps:.2f}", # The fast number
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
            )
            cv2.putText(
                enhanced_cv,
                f"System FPS: {loop_fps:.2f}", # The "realistic" slow number
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 3 # Different color
            )

            # === Convert and Update View ===
            _, buffer = cv2.imencode('.jpg', enhanced_cv)
            enhanced_b64_display = b64encode(buffer).decode('utf-8')
            eval_js(f"updateImageViews('{original_b64_display}', '{enhanced_b64_display}')")

    except KeyboardInterrupt:
        print("\nLoop stopped by user.")
    except Exception as e:
        print(f"Loop stopped due to an error: {e}")
    finally:
        print("Attempting to stop webcam stream...")
        eval_js('stopCamera()')
        print("Webcam stream has been stopped.")

# --- Run the main demo function ---
live_side_by_side_demo()


# In[ ]:


# ==============================================================================
#      FINAL LIVE SIDE-BY-SIDE DEMO (Smoothed FPS Version)
# ==============================================================================

from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
import cv2
import numpy as np
import time
from torchvision import transforms
import io

# We assume 'model', 'DEVICE', and other necessary variables exist from setup.
# ------------------------------------------------------------------------------

def pil_to_b64(pil_img):
    """Converts a PIL Image to a base64 encoded string for HTML display."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return b64encode(buffered.getvalue()).decode('utf-8')

def live_side_by_side_demo():
    """
    Main function to run the entire live demo with side-by-side comparison
    and smoothed FPS calculation.
    """
    # --- 1. Define all necessary JavaScript and HTML ---
    js_code = """
    var video;
    var stream;
    async function setupLayoutAndCamera() {
      video = document.createElement('video');
      video.id = 'camera-video';
      video.setAttribute('playsinline', '');
      video.style.display = 'none';
      const div = document.createElement('div');
      div.appendChild(video);
      document.body.appendChild(div);
      stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream;
      await video.play();
      return "Camera is ready!";
    }
    async function captureFrame() {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    function updateImageViews(original_b64, enhanced_b64) {
      document.getElementById('original-view').src = 'data:image/jpeg;base64,' + original_b64;
      document.getElementById('enhanced-view').src = 'data:image/jpeg;base64,' + enhanced_b64;
    }
    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    }
    """

    html_layout = """
    <div style="display: flex; justify-content: space-around; align-items: center;">
      <div style="text-align: center;">
        <h3>Original Feed</h3>
        <img id="original-view" width="480">
      </div>
      <div style="text-align: center;">
        <h3>Enhanced Feed (Model FPS + System FPS)</h3>
        <img id="enhanced-view" width="480">
      </div>
    </div>
    """

    # --- 2. Inject JS/HTML and start the camera (The CORRECT SINGLE method) ---
    display(HTML(html_layout))
    display(Javascript(js_code))
    print("Initializing webcam... Please allow camera access.")
    try:
        eval_js('setupLayoutAndCamera()')
        print("✅ Webcam is active. Starting real-time enhancement loop...")
    except Exception as e:
        print(f"❌ Could not start camera. Error: {e}")
        return

    # --- 3. Run the main enhancement loop ---
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    MODEL_INPUT_SIZE = (480, 270)

    try:
        # --- FPS TRACKING SETUP ---
        frame_count = 0
        fps_update_interval = 1.0  # Update the System FPS display every 1 second
        last_fps_time = time.time()
        smoothed_fps = 0

        while True:
            # A. Capture Frame
            b64_data_raw = eval_js('captureFrame()')
            original_pil = PIL.Image.open(io.BytesIO(b64decode(b64_data_raw.split(',')[1])))

            # B. Preprocess (FIX #2: Add PIL.Image.BICUBIC)
            input_tensor = to_tensor(original_pil.resize(MODEL_INPUT_SIZE, PIL.Image.BICUBIC)).unsqueeze(0).to(DEVICE)

            # C. Model Inference + Model FPS Timing
            model_start = time.time()
            with torch.no_grad():
                output_tensor = model(input_tensor)
            model_time = time.time() - model_start
            model_fps = 1.0 / model_time if model_time > 0 else 0

            # D. Postprocess
            enhanced_pil = to_pil(output_tensor.clamp(0, 1).squeeze(0).cpu())
            enhanced_cv = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)

            # E. Rolling FPS Update
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= fps_update_interval:
                smoothed_fps = frame_count / (current_time - last_fps_time)
                last_fps_time = current_time
                frame_count = 0

            # F. Overlay both FPS on output
            cv2.putText(enhanced_cv, f"Model FPS: {model_fps:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(enhanced_cv, f"System FPS: {smoothed_fps:.2f}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 140, 255), 3)

            # G. Update display
            _, buffer = cv2.imencode('.jpg', enhanced_cv)
            enhanced_b64_display = b64encode(buffer).decode('utf-8')
            original_b64_display = pil_to_b64(original_pil)

            eval_js(f"updateImageViews('{original_b64_display}', '{enhanced_b64_display}')")

    except KeyboardInterrupt:
        print("\n⏹️ Loop stopped by user.")
    except Exception as e:
        print(f"❌ Loop stopped due to error: {e}")
    finally:
        # H. Clean up and stop the camera
        print("Attempting to stop webcam stream...")
        eval_js('stopCamera()')
        print("🛑 Webcam stream has been stopped.")


# --- FIX #3: Call the main function to run it ---
live_side_by_side_demo()

