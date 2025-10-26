/**
 * This function runs after HTML loads
 * It waits for Run Prediction button to be pressed
 * When pressed it connects to a locally hosted flask server hosting nural network
 */


//DOMContentLoaded waits for HTML to load before proceeding
document.addEventListener('DOMContentLoaded', () => {
  const fetchButton = document.getElementById('fetch-button');
  const resultDisplay = document.getElementById('result-display');
  const urlDisplay = document.getElementById('url-display');

  // Add a click listener to the button
  // Gets data from server on click
  fetchButton.addEventListener('click', () => {
    fetchDataFromServer();
  });

  // Fetches Data from Flask Server
  // Added async to prevent program waiting for data to be fetched
  async function fetchDataFromServer() {
    resultDisplay.textContent = 'Contacting server...';

    //Code for getting the URL using chrome.tabs API
    const tabs = await chrome.tabs.query({active: true, currentWindow: true});
    const currentTabUrl = tabs[0].url; //Gets url from tabs

    try {
      // Make the API call to the server
      // Call server route /get-data
      const response = await fetch('http://127.0.0.1:5000/get-data', {
          //Below sends the url to the server as a json
          method: 'Post',
          headers: {
              'content-Type': 'application/json'
          },
          body: JSON.stringify({url: currentTabUrl})
      });

      // Check if the server responded successfully
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      // Get the JSON data from the response
      const data = await response.json();

      // Display the message from the server
      //urlDisplay.textContent = data.site_url_used;
      resultDisplay.textContent = data.message;

    } catch (error) {
      // This will run if server cannot be reached
      console.error('Error fetching data:', error);
      resultDisplay.textContent = 'Error: Could not connect. Is the server running?';
    }
  }
});
