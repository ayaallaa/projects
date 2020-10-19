var randomScalingFactor = function(){ return Math.round(Math.random()*1000)};
var date = new Date();
//var Accuracy=0;
//console.log(date.getDate());
var firebaseConfig = {
    apiKey: "AIzaSyDTwwK3SlpVUHnex6xCkEbvKIZSGOCYghY",
    authDomain: "speechmania-b547e.firebaseapp.com",
    databaseURL: "https://speechmania-b547e.firebaseio.com",
    projectId: "speechmania-b547e",
    storageBucket: "speechmania-b547e.appspot.com",
    messagingSenderId: "694335410890",
    appId: "1:694335410890:web:8d333835b9433c781b9a33",
    measurementId: "G-GZFRM55DSZ"
  };
  firebase.initializeApp(firebaseConfig);
  /*var storageRef = firebase.storage().ref();
 // Create a reference to the file whose metadata we want to retrieve
var forestRef = storageRef.child('audios/fromUnity.wav');
console.log(forestRef.getMetadata(location));
//getMetadataForFileList(forestRef)
forestRef.getMetadata(location).then(function(metadata) {
	// Metadata now contains the metadata for 'images/forest.jpg'
  }).catch(function(error) {
	// Uh-oh, an error occurred!
  });

  function getMetadataForFileList(forestRef) {
	for (const file of forestRef) {
	  // Not supported in Safari for iOS.
	  const name = file.name ? file.name : 'NOT SUPPORTED';
	  // Not supported in Firefox for Android or Opera for Android.
	  const type = file.type ? file.type : 'NOT SUPPORTED';
	  console.log(type);
	  // Unknown cross-browser support.
	  const size = file.size ? file.size : 'NOT SUPPORTED';
	  console.log({file, name, type, size});
	}
  }*/



  var old_chair_accuracy=0;
  var old_door_accuracy=0;
  var chart1 = document.getElementById("line-chart").getContext("2d");
  
  var y=[];
  var ynew=[];
  var z=[];
  var x=[] /*Array.from(Array(10000).keys()).map(i => 0 + i *5 )*/;
  i=1;
  // Initialize Firebase
  
 //piechart(Accuracy);
  var msglist = document.getElementById("easypiechart-blue");
  //firebase.analytics();
  var ref = firebase.database().ref().child('speechmania-b547e');
 
  ref.on("value", function(snapshot) {
     var Data=snapshot.val();
     var chair_accuracy=(Data.Chair_Accuracy);
	 var door_accuracy=(Data.Door_Accuracy);
     var apple_accuracy=(Data.Apple_Accuracy);
	 var tired_accuracy=(Data.Tired_Accuracy);
	 var water_accuracy=(Data.Water_Accuracy);
	 var juice_accuracy=(Data.Juice_Accuracy);
	 var eat_accuracy=(Data.Eat_Accuracy);
	 var tree_accuracy=(Data.Tree_Accuracy);
	 var taxi_accuracy=(Data.Taxi_Accuracy);
	 var pound_accuracy=(Data.Pound_Accuracy);

	 //for the actual word and predicted word:
     var actual_word=(Data.Actual_Word);
	 var predicted_word=(Data.Predicted_Word);
	 console.log(actual_word);
	 console.log(predicted_word);
//for pie chart
	document.getElementById('easypiechart-blue').getElementsByClassName('percent')[0].innerHTML=String(chair_accuracy)+'%';
	document.getElementById('easypiechart-orange').getElementsByClassName('percent')[0].innerHTML=String(door_accuracy)+'%';
	document.getElementById('easypiechart-teal').getElementsByClassName('percent')[0].innerHTML=String(apple_accuracy)+'%';
	document.getElementById('easypiechart-red').getElementsByClassName('percent')[0].innerHTML=String(tired_accuracy)+'%';
	/*document.getElementById('easypiechart-green').getElementsByClassName('percent')[0].innerHTML=String(water_accuracy)+'%';
	document.getElementById('easypiechart-black').getElementsByClassName('percent')[0].innerHTML=String(juice_accuracy)+'%';
	document.getElementById('easypiechart-yellow').getElementsByClassName('percent')[0].innerHTML=String(eat_accuracy)+'%';
	document.getElementById('easypiechart-blue2').getElementsByClassName('percent')[0].innerHTML=String(tree_accuracy)+'%';*/

	console.log(Data);
	console.log(chair_accuracy);
	console.log(door_accuracy);
    if ((chair_accuracy!=old_chair_accuracy)||(door_accuracy!=old_door_accuracy))
    {
		//for the comparison between actual word and predicted word:
		if (actual_word==predicted_word)
		{
		  y.push(Data.Chair_Accuracy);
		  z.push(Data.Door_Accuracy);
		  x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i)*/);
		  createChart(x, y);
		  createChart(x, z);
		  old_chair_accuracy=chair_accuracy;
		  old_door_accuracy=door_accuracy;
		  i=i+1;
		}
		else{
			y.push(0);
			z.push(0);
			x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i)*/);
		  createChart(x, y);
		  createChart(x, z);
		}

		


	 /* z.push(Data.Door_Accuracy);
	  y.push(Data.Chair_Accuracy);
	  x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i));
	  createChart(x, y);
	  createChart(x, z);
	  old_chair_accuracy=chair_accuracy;
	  old_door_accuracy=door_accuracy;
	  i=i+1;*/
    }
});

var lineChartData = {
	labels : x,
	datasets : [
		{
			label: "Chair",
			fillColor : "rgba(220,220,220,0.2)",
			strokeColor : "rgba(220,220,220,1)",
			pointColor : "rgba(220,220,220,1)",
			pointStrokeColor : "#fff",
			pointHighlightFill : "#fff",
			pointHighlightStroke : "rgba(220,220,220,1)",
			data : y
		},
		{
			label: "Door",
			fillColor : "rgba(48, 164, 255, 0.2)",
			strokeColor : "rgba(48, 164, 255, 1)",
			pointColor : "rgba(48, 164, 255, 1)",
			pointStrokeColor : "#fff",
			pointHighlightFill : "#fff",
			pointHighlightStroke : "rgba(48, 164, 255, 1)",
			data : z
		}
	]

}



function createChart(x,y){
	
	window.myLine = new Chart(chart1).Line(lineChartData, {
	responsive: true,
	scaleLineColor: "rgba(0,0,0,.2)",
	scaleGridLineColor: "rgba(0,0,0,.05)",
	scaleFontColor: "#c5c7cc"
	});};











var pieData = [
	{
		value: 50,
		color:"#A84848",
		highlight: "#A84848",
		label: "Blue"
	},
	{
		value: 100,
		color: "#ffb53e",
		highlight: "#fac878",
		label: "Orange"
	},
	{
		value: 100,
		color: "#1ebfae",
		highlight: "#3cdfce",
		label: "Teal"
	},
	{
		value: 220,
		color: "#f9243f",
		highlight: "#f6495f",
		label: "Red"
	},


	{
		value: 50,
		color:"#A84848",
		highlight: "#A84848",
		label: "Blue"
	},
	{
		value: 100,
		color: "#ffb53e",
		highlight: "#fac878",
		label: "Orange"
	},
	{
		value: 100,
		color: "#1ebfae",
		highlight: "#3cdfce",
		label: "Teal"
	},
	{
		value: 220,
		color: "#4F0B27",
		highlight: "#4F0B27",
		label: "green"
	}

];













	var barChartData = {
		labels : ["January","February","March","April","May","June","July"],
		datasets : [
			{
				fillColor : "rgba(220,220,220,0.5)",
				strokeColor : "rgba(220,220,220,0.8)",
				highlightFill: "rgba(220,220,220,0.75)",
				highlightStroke: "rgba(220,220,220,1)",
				data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
			},
			{
				fillColor : "rgba(48, 164, 255, 0.2)",
				strokeColor : "rgba(48, 164, 255, 0.8)",
				highlightFill : "rgba(48, 164, 255, 0.75)",
				highlightStroke : "rgba(48, 164, 255, 1)",
				data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
			}
		]

	}

			
	var doughnutData = [
				{
					value: 300,
					color:"#30a5ff",
					highlight: "#62b9fb",
					label: "Blue"
				},
				{
					value: 50,
					color: "#ffb53e",
					highlight: "#fac878",
					label: "Orange"
				},
				{
					value: 100,
					color: "#1ebfae",
					highlight: "#3cdfce",
					label: "Teal"
				},
				{
					value: 120,
					color: "#f9243f",
					highlight: "#f6495f",
					label: "Red"
				}

			];
			
	var radarData = {
	    labels: ["Eating", "Drinking", "Sleeping", "Designing", "Coding", "Cycling", "Running"],
	    datasets: [
	        {
	            label: "My First dataset",
	            fillColor: "rgba(220,220,220,0.2)",
	            strokeColor: "rgba(220,220,220,1)",
	            pointColor: "rgba(220,220,220,1)",
	            pointStrokeColor: "#fff",
	            pointHighlightFill: "#fff",
	            pointHighlightStroke: "rgba(220,220,220,1)",
	            data: [65, 59, 90, 81, 56, 55, 40]
	        },
	        {
	            label: "My Second dataset",
	            fillColor : "rgba(48, 164, 255, 0.2)",
	            strokeColor : "rgba(48, 164, 255, 0.8)",
	            pointColor : "rgba(48, 164, 255, 1)",
	            pointStrokeColor : "#fff",
	            pointHighlightFill : "#fff",
	            pointHighlightStroke : "rgba(48, 164, 255, 1)",
	            data: [28, 48, 40, 19, 96, 27, 100]
	        }
	    ]
	};
	
	var polarData = [
		    {
		    	value: 300,
		    	color: "#1ebfae",
		    	highlight: "#38cabe",
		    	label: "Teal"
		    },
		    {
		    	value: 140,
		    	color: "#ffb53e",
		    	highlight: "#fac878",
		    	label: "Orange"
		    },
		    {
		    	value: 220,
		    	color:"#30a5ff",
		    	highlight: "#62b9fb",
		    	label: "Blue"
		    },
		    {
		    	value: 250,
		    	color: "#f9243f",
		    	highlight: "#f6495f",
		    	label: "Red"
		    }
		
	];

