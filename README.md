# ChatGPT prompt samples (OpenAI + GitHub Copilot)

## 1. Bridge: simulation of 3 hands dealing 
___
&emsp;Another day I was arguing with my friends about how to bid for a given hand in bridge.<br>
&emsp;I'm in 3rd position and the bidding was: _Pas, Pas_<br>
&emsp;My hand:<br>
<div style="display: flex; direction: row-reverse;">
    <img style="width: 100px; position: relative; left: 0px;" 
        src="https://upload.wikimedia.org/wikipedia/commons/f/f4/Ace_of_spades2.svg" 
        alt="Ace of spades">
    <img style="width: 100px; position: relative; left: -40px; top:0" 
        src="https://upload.wikimedia.org/wikipedia/commons/e/ee/King_of_spades.svg" 
        alt="King of spades">
    <img style="width: 100px; position: relative; left: -80px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Queen_of_spades.svg" 
        alt="Queen of spades">
    <img style="width: 100px; position: relative; left: -120px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/6/68/10_of_spades.svg" 
        alt="10 of spades">
    <img style="width: 100px; position: relative; left: -160px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/f/f7/7_of_spades.svg" 
        alt="7 of spades">
    <img style="width: 100px; position: relative; left: -200px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/8/8a/5_of_spades.svg" 
        alt="5 of spades">
    <img style="width: 100px; position: relative; left: -240px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/e/eb/3_of_spades.svg" 
        alt="3 of spades">
    <img style="width: 100px; position: relative; left: -280px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/9/9d/9_of_hearts.svg" 
        alt="9 of hearts">
    <img style="width: 100px; position: relative; left: -320px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/7/7e/6_of_hearts.svg" 
        alt="6 of hearts">
    <img style="width: 100px; position: relative; left: -360px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/e/e9/4_of_hearts.svg" 
        alt="4 of hearts">
    <img style="width: 100px; position: relative; left: -400px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/d/db/7_of_clubs.svg" 
        alt="7 of clubs">
    <img style="width: 100px; position: relative; left: -440px; top:0"  
        src="https://upload.wikimedia.org/wikipedia/commons/7/72/5_of_clubs.svg" 
        alt="5 of clubs">
    <img style="width: 100px; position: relative; left: -480px; top:2"  
        src="https://upload.wikimedia.org/wikipedia/commons/5/5a/8_of_diamonds.svg" 
        alt="8 of diamonds">

</div>

<!-- from https://commons.wikimedia.org/w/index.php?search=Byron+Knoll+Playing+cards&title=Special:MediaSearch&go=Go&type=image&sort=recency -->


&emsp;If I had had less than 6-7 points, I would have clearly bid 3 spades, but with 9 points and a nice distribution, I thought that maybe I would go down 3 but could make 2. 😁
Or maybe they don't have a game, and I can push myself up to 3 spades in the bidding or even 4s if they discover they have a game in hearts.<br>
&emsp;They argued that after the two "Pass", the chance that the opponents have >=24 points is high, so it is not good to give them time to discover their suits; therefore, I should bid right away: 3 spades.<br>
&emsp;But my gut didn't really tell me that they were more likely to have game points…😄

&emsp;Where does this simulation come from 😉<br>
&emsp;(honestly, it was also an excuse to see how I get along with the ChatGPT prompt and GitHub Copilot in VS) 

...and the results:<br>(via Pyodide terminal emulator / Python 3.11.3 (main, Mar 31 2024 11:27:51) on WebAssembly/Emscripten 👍)
![Bridge simulation of 3 hands dealing ](Bridge/BridgeOnPyodide.png)