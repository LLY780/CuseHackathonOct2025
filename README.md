Push and Pull project files for CuseHacks Hackathon 2025
# Setup

## How to Install Required Packages
Open your terminal (CMD, PowerShell, or Terminal on Mac).

Navigate (cd) into the project's main folder.

Run the following command to install all required Python libraries:

pip install -r requirements.txt

## How to Download the Model
Open the following google drive link: https://drive.google.com/drive/folders/1y7pyStyPvBOWNnyIcH1Qu1YADEhUqxR3?usp=drive_link

Download the model folder

Place the zipped folder in the same directory as read.py

Unzip the folder

## How to Set Up the Chrome Extension (Frontend)
Open Google Chrome, type chrome://extensions into the address bar, and press Enter.

Find the "Developer mode" toggle (usually in the top-right corner) and turn it on.

Click the "Load unpacked" button.

In the file dialog that opens, select the entire project folder (the one containing manifest.json).

Click "Select Folder". The extension will now be installed and appear in your toolbar.

# Ways to Use the Code
## Using the chrome extension (Recommeded)
Download the server.py and chromeExtension folder
Follow chrome extension setup instructions above
The server.py file and chromeExtension folder must be in the same directory as read.py
Run server py, this is what connects to the extension and must be running for it work

## Training the classification model
You can train the model for yourself following the steps below.

Download and run the train.py python file
The file will create a model folder in the current directory
This model folder will be used for reading text and url
The folder must be in the same directory as read.py

## Reading text and getting predictions
To simply read a text file follow the steps below

Download and run the read.py and scrape.py python files
Run the read.py file using command line arguments
read.py takes either a article url or string of text
The file prints out the probability distribution


  
