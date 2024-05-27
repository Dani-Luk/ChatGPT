<style>
.container {
  position: relative;
  top: 0px;
  width: 120px; /* Adjust as needed */
  height: 200px;
}

.image1 {
  position: absolute;
  /* top: 0px; */
  left: 0px;
  /*width: 200px; /* Adjust as needed */
  /* height: auto; */
}

.image2 {
  position: absolute;
  /* top: 0px; Adjust as needed */
  left: 70px; /* Adjust as needed */
  /* width: 200px; Adjust as needed */
  /* height: auto; */
  
}
</style>

<div class="container">
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/f4/Ace_of_spades2.svg" alt="Image 1" class="image1">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Queen_of_spades.svg"  alt="Image 2" class="image2">
</div>


<div style="position: relative; width: 200px; height: 200px;">
  <img src="https://via.placeholder.com/200x150.png?text=Image+1" alt="Image 1" style="position: absolute; top: 0; left: 0; width: 200px;">
  <img src="https://via.placeholder.com/200x150.png?text=Image+2" alt="Image 2" style="position: absolute; top: 50px; left: 50px; width: 200px;">
</div>

