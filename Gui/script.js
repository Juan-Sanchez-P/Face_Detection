function uploadVideo() {
    let fileInput = document.getElementById("videoUpload");
    let file = fileInput.files[0];
    
    if (file) {
        let videoElement = document.getElementById("videoPreview");
        videoElement.src = URL.createObjectURL(file);
        videoElement.load();
        
        document.getElementById("status").innerText = "Processing...";
        
        let reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            eel.process_video(reader.result)(updateUI);
        };
    } else {
        alert("Please select a video.");
    }
}

function updateUI(result) {
    document.getElementById("status").innerText = "Processing Complete!";
    
    let objectList = document.getElementById("objectList");
    objectList.innerHTML = "";
    
    result.forEach(item => {
        let li = document.createElement("li");
        li.innerText = `${item.label} - Confidence: ${item.confidence}%`;
        objectList.appendChild(li);
    });
}
