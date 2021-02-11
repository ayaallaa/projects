//Global Variables 
let d = new Date();
let newDate = d.getMonth()+'.'+ d.getDate()+'.'+ d.getFullYear();
const apikey="773c64a43a6e3ff480858ddac91dd622"
let feelings ;
let zipcode;
//="94040" ;
let temp;
let url;
const temphtml=document.getElementById("temp");
const datehtml=document.getElementById("date");
const feelingshtml=document.getElementById("content");



const generate=document.getElementById("generate");

generate.addEventListener('click', function () {
  feelings=document.getElementById("feelings").value;
  zipcode=document.getElementById("zip").value;
  console.log(feelings);
  console.log(zipcode);
  url="http://api.openweathermap.org/data/2.5/weather?zip="+zipcode+"&appid=773c64a43a6e3ff480858ddac91dd622&units=metric";
  console.log(url);
  generateNow();  
});


const getTemp = async (url)=>{

  const res = await fetch(url)
  try {

    const data = await res.json();
    //console.log(data)
    temp=data.main.temp;
    //console.log(temp);
    return data;
  }  catch(error) {
    console.log("error", error);
      //alert(error);
    // appropriately handle the error
  }
}



function generateNow(){

   getTemp(url)
   .then(function(data){
   // console.log(data)
    postData('/postdata',{date:newDate, temp:temp, feelings:feelings})
    
    })
    //updateUI
    .then(()=>view())

};



/* get weather information from server */

const postData = async ( url='', data = {})=>{

    const response = await fetch(url, {
    method: 'POST', 
    credentials: 'same-origin', 
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),       
})

    try {
    const newContent = await response.json();
    console.log(newContent)
            return newContent;
    }catch(error) {
    console.log('error', error);
    
    }
}

//updateUI
const view = async () => {
    const request = await fetch('/all');
    try{
    const allData = await request.json()
    console.log(allData);
temphtml.innerHTML  = "temp=" + allData.temp ;   
datehtml.innerHTML = "date is " +allData.date ; 
feelingshtml.innerHTML ="i feel " +allData.feelings ; 
}
catch(error){
    console.log("error", error);
    }
}
