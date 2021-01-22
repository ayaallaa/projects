/**
 * 
 * Manipulating the DOM exercise.
 * Exercise programmatically builds navigation,
 * scrolls to anchors from navigation,
 * and highlights section in viewport upon scrolling.
 * 
 * Dependencies: None
 * 
 * JS Version: ES2015/ES6
 * 
 * JS Standard: ESlint
 * 
*/
/*eslint-env es6*/
/**
 * Define Global Variables
 * 
*/
const navul = document.getElementById("navbar__list");
const fragment_nav = document.createDocumentFragment();
const sections =document.getElementsByClassName("landing__container");


// build the nav
for (const section of sections) {
    let a = document.createElement('a');
   let li = document.createElement('li');
    let text=section.firstElementChild.textContent;
    li.textContent = text  ;
    a.appendChild(li);
    fragment_nav.appendChild(a);

}
navul.appendChild(fragment_nav);
const navchilds=navul.childNodes;




// Build menu 
const sec1= document.getElementById("section1");
let sec1top=sec1.offsetTop;
const sec2= document.getElementById("section2");
let sec2top=sec2.offsetTop;
const sec3= document.getElementById("section3");
let sec3top=sec3.offsetTop;
const sec4= document.getElementById("section4");
let sec4top=sec4.offsetTop;

// Scroll to section on link click
navchilds[0].addEventListener('click', function () {
  window.scrollTo(0,sec1top );
});
navchilds[1].addEventListener('click', function () {
  window.scrollTo(0,sec2top );
});
navchilds[2].addEventListener('click', function () {
  window.scrollTo(0,sec3top );
});
navchilds[3].addEventListener('click', function () {
  window.scrollTo(0,sec4top );
});


// Set sections as active
let activeSec =navul;
function activeSection (){

for (const section of sections) {
    let topSec = section.getBoundingClientRect().top; 
    //console.log(section);
    //console.log(topSec);
    if(topSec >= 0 && topSec <= 250)
        {
            activeSec=section;
            //console.log(activeSec);
        }
    section.style.background= 'linear-gradient(0deg, rgba(255,255,255,.1) 0%, rgba(255,255,255,.2) 100%)';
    }
    
activeSec.style.background = '#2B60DE';
  
}


//window.addEventListene ‘scroll’,
window.addEventListener('scroll', function () {
  activeSection();
  //console.log(sec3Top)
});



//top button and scroll to top
const Btop = document.createElement('a');
const foot =document.getElementsByClassName('page__footer')[0] ;
Btop.innerHTML = '<button type="button">Top</button>';
foot.appendChild(Btop);
Btop.addEventListener('click', function () {
  window.scrollTo(0, 0);
});


