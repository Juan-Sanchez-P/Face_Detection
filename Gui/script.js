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
            eel.process_video(reader.result, "all")(updateUI);
        };
    } else {
        alert("Please select a video.");
    }
}

function removeVideo() {
    let videoElement = document.getElementById("videoPreview");
    let fileInput = document.getElementById("videoUpload");
    let statusText = document.getElementById("status");
    let objectList = document.getElementById("objectList");

    videoElement.src = "";
    fileInput.value = "";
    statusText.innerText = "";
    objectList.innerHTML = "";
}

function detectObject(objectType) {
    let fileInput = document.getElementById("videoUpload");
    let file = fileInput.files[0];
    
    if (file) {
        document.getElementById("status").innerText = "Detecting " + objectType + "...";
        
        let reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            eel.process_video(reader.result, objectType)(updateUI);
        };
    } else {
        alert("Please upload a video first.");
    }
}

function updateUI(result) {
    document.getElementById("status").innerText = "Processing Complete!";
    
    let objectList = document.getElementById("objectList");
    objectList.innerHTML = "";
    
    if (result.length === 0) {
        objectList.innerHTML = "<li>No objects detected.</li>";
    }
    
    result.forEach(item => {
        let li = document.createElement("li");
        li.innerText = `${item.label} - Confidence: ${item.confidence}%`;
        objectList.appendChild(li);
    });
}

document.getElementById("dogButton").onclick = function() { detectObject("dog"); };
document.getElementById("catButton").onclick = function() { detectObject("cat"); };
document.getElementById("carButton").onclick = function() { detectObject("car"); };
document.getElementById("humanButton").onclick = function() { detectObject("human"); };
document.getElementById("bicycleButton").onclick = function() { detectObject("bicycle"); };
document.getElementById("chairButton").onclick = function() { detectObject("chair"); };
