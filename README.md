# SmartWasteNet: AI-Powered Waste Detection & Analytics

SmartWasteNet is an integrated AI system designed to automate waste classification and provide actionable environmental analytics for smart cities. By combining computer vision with sustainability tracking, the platform helps optimize recycling efficiency and monitor environmental impact.

🚀 Key Features

    AI Waste Detection: Real-time classification of waste into categories: Plastic, Glass, Metal, and Other using a custom-trained YOLOv8 model.

    Recycling Advice: Provides specific instructions on how to handle detected waste (e.g., "Rinse the glass and remove lids").

    Sustainability Analytics: Dashboard tracking total waste, recycling efficiency (percentage of recyclable vs. non-recyclable), and estimated CO₂ savings.

    Circular Economy Insights: AI-generated suggestions for urban waste management based on local waste composition.

    Model Performance Tracking: Deep-dive analytics into training metrics, including precision-recall curves and confusion matrices.

🛠️ Tech Stack

    AI/ML Framework: Ultralytics YOLOv8.

    Dashboard: Streamlit.

    Data Processing: Pandas, NumPy, and JSON.

    Visualization: Matplotlib and Streamlit Line/Bar Charts.

    Programming Language: Python.

# 📂 Project Structure
```plaintext
SmartWasteNet/
├── app.py                # Main Streamlit dashboard application
├── configs/
│   └── dataset.yaml      # YOLOv8 dataset configuration
├── model/
│   └── best.pt           # Trained weights for waste detection
├── outputs/
│   └── analytics.json    # Local storage for waste statistics
├── scripts/
│   ├── convert_dataset.py # Script to format images into YOLO format
│   └── train.py          # Script for model training
├── utils/
│   ├── analytics.py      # Logic for updating waste statistics
│   ├── helpers.py        # Sustainability scoring data
│   └── recycling_info.py # Recycling tips and environmental impact data
└── requirements.txt      # Project dependencies
```

⚙️ Installation & Setup

    Clone the Repository:
    Bash

    git clone https://github.com/your-username/SmartWasteNet.git
    cd SmartWasteNet

    Install Dependencies:
    Bash

    pip install -r requirements.txt

    Prepare the Model:
    Ensure your trained weights file is located at model/best.pt.

    Run the Application:
    Bash

    streamlit run app.py

🧠 Training Your Own Model

If you wish to retrain the model on your own dataset:

    Place your raw images in dataset/images/.

    Run the conversion script to generate YOLO-compatible labels:
    Bash

    python scripts/convert_dataset.py

    Execute the training script:
    Bash

    python scripts/train.py

    The training script is configured for 40 epochs and will automatically use a GPU if available.

♻️ Environmental Impact Calculation

The project calculates CO₂ savings based on the weight of material diverted from landfills:

    Plastic: 6kg CO₂ saved/unit

    Metal: 9kg CO₂ saved/unit

    Glass: 4kg CO₂ saved/unit

📜 License

This project is for educational and environmental sustainability purposes. See LICENSE for more details.
