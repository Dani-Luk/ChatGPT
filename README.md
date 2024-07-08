# ChatGPT prompt samples (OpenAI + GitHub Copilot)

## 1. Bridge: Simulation of Dealing 3 Hands
___
&emsp;Another day I was arguing with my friends about how to bid for a given hand in bridge.<br>
&emsp;I'm in 3rd position and the bidding was: _Pas, Pas_<br>
&emsp;My hand:<br>
<div style="display: flex; direction: row-reverse;">
    <img style="width: 80px; position: relative; left: 0px;" 
        src="https://upload.wikimedia.org/wikipedia/commons/f/f4/Ace_of_spades2.svg" 
        alt="Ace of spades">
    <img style="width: 80px; position: relative; left: -40px; top:0" 
        src="https://upload.wikimedia.org/wikipedia/commons/e/ee/King_of_spades.svg" 
        alt="King of spades">
    <img style="width: 80px; position: relative; left: -80px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Queen_of_spades.svg" 
        alt="Queen of spades">
    <img style="width: 80px; position: relative; left: -120px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/6/68/10_of_spades.svg" 
        alt="10 of spades">
    <img style="width: 80px; position: relative; left: -160px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/f/f7/7_of_spades.svg" 
        alt="7 of spades">
    <img style="width: 80px; position: relative; left: -200px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/8/8a/5_of_spades.svg" 
        alt="5 of spades">
    <img style="width: 80px; position: relative; left: -240px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/e/eb/3_of_spades.svg" 
        alt="3 of spades">
    <img style="width: 80px; position: relative; left: -280px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/9/9d/9_of_hearts.svg" 
        alt="9 of hearts">
    <img style="width: 80px; position: relative; left: -320px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/7/7e/6_of_hearts.svg" 
        alt="6 of hearts">
    <img style="width: 80px; position: relative; left: -360px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/e/e9/4_of_hearts.svg" 
        alt="4 of hearts">
    <img style="width: 80px; position: relative; left: -400px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/d/db/7_of_clubs.svg" 
        alt="7 of clubs">
    <img style="width: 80px; position: relative; left: -440px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/7/72/5_of_clubs.svg" 
        alt="5 of clubs">
    <img style="width: 80px; position: relative; left: -480px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/5/5a/8_of_diamonds.svg" 
        alt="8 of diamonds">

</div>

<!-- from https://commons.wikimedia.org/w/index.php?search=Byron+Knoll+Playing+cards&title=Special:MediaSearch&go=Go&type=image&sort=recency -->


&emsp;If I had had less than 6-7 points, I would have clearly bid 3 spades, but with 9 points and a nice distribution, I thought that maybe I would go down 3 but could make 2. 😁
Or maybe they don't have a game, and I can push myself up to 3 spades in the bidding or even 4s if they discover they have a game in hearts.<br>
&emsp;They argued that after the two "Pass", the chance that the opponents have >=24 points is high, so it is not good to give them time to discover their suits; therefore, I should bid right away: 3 spades.<br>
&emsp;But my gut didn't really tell me that they were more likely to have game points…😄

&emsp;Where does this simulation come from.😉

_**NOTE:** Actually, it was more of an excuse to see how I get along with the ChatGPT prompt and GitHub Copilot in VS Code. You can view the initial prompt [here](<Bridge/Bridge.ChatGPT 4o.txt>) and the subsequent ones in the commits history of bridge.py file._

And the results:<br>(via Pyodide terminal emulator / Python 3.11.3 (main, Mar 31 2024 11:27:51) on WebAssembly/Emscripten 👍)
![Bridge simulation of 3 hands dealing ](Bridge/BridgeOnPyodide.png)

## 2. NN-XOR Solved by 2 Crossing Layers (2 x 1 Perceptron)
___
&emsp;The same old story of the XOR problem😁, now solved by a 2-layer model with each layer having 1 perceptron.<br>
&emsp;The first layer will learn the bitwise AND operation (or something else, at your discretion) and try to best separate its classes.<br>
&emsp;The second layer, by incorporating the output of the first (frozen) layer as a third input dimension(z-elevation), will attempt to achieve the XOR results.

&emsp;You can start training the second layer at any point in the learning process of the first layer simply by moving the cursor of the first slider and clicking the "Train layer 2" button.<br>

The second plot shows:
- the 4 input data points(00, 01, 10, 11) corroborated with z-elevation given by the first model's result
- ${\color{#f7fc00}■}$ the decision boundary plane
- ${\color{#96e8e8}■}$ the z=0 plane 
- ${\color{#81bf3d}■}$ the line of intersection of this two planes 

The circles (points) on the plots are colored in such a way that: 
- there are 2 colors for the border, representing the 2 classes
<img style="width: 15px; hight: 15px"  
        src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Eo_circle_red_circle.svg/240px-Eo_circle_red_circle.svg.png" 
        alt="">
<img style="width: 15px; hight: 15px"  
        src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Eo_circle_blue_circle.svg/240px-Eo_circle_blue_circle.svg.png" 
        alt="">

- there are 2 colors (the same ${\color{red}●}$ ${\color{blue}●}$ for an easier visualization) for the inside, representing the side position relative to the decision boundary (line or plane).<br>

The model is considered 'good' if the points on one side have the same color pattern.

#### A snapshot:
<!-- ![png snapshot](<NN-XOR cross 2 layers/NN-XOR cross 2 layers.png>) -->
[![png snapshot][IMG_PNG]][IMG_PNG]

#### An animated Gif (thx [Ezgif](https://ezgif.com/)):
[![Gif animated][ANIM_GIF]][ANIM_GIF]

#### YouTube video: [https://youtu.be/St4yNx8MQJA](https://youtu.be/St4yNx8MQJA)


[IMG_PNG]: <NN-XOR cross 2 layers/NN-XOR cross 2 layers.png>
[ANIM_GIF]: <NN-XOR cross 2 layers/NN-XOR cross 2 layers 10 sec 895 px.gif>