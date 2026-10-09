\# CREST Final YOLO26 Vision System



\## 1. Purpose



This folder contains the finalized YOLO26-based CREST vision system used for real-time pallet tracking and pallet-state classification on the conveyor.



The final system detects and tracks pallets and classifies each pallet as:



\- EMPTY

\- ACCEPTED

\- REJECTED



Final decision logic:



\- EMPTY pallet -> EMPTY

\- Object present with X mark -> REJECTED

\- Object present without X / accepted mark -> ACCEPTED



The system also maintains the previous reliable pallet state during temporary occlusions such as a hand, wire, or other object passing in front of the pallet.



\---



\## 2. Final Folder Structure



C:\\Do\_Not\_Delete\_PLC\\finalv26



CODE

\- live\_final\_crest\_yolo26.py

\- dynamic\_live\_pallet\_detection.py



CONFIG

\- conveyor\_boundary.json



MODELS

\- pallet\_yolo26n\_best.pt

\- empty\_object\_yolo26n\_cls\_best.pt

\- x\_no\_x\_yolo26n\_medium\_focus\_final\_hard\_best.pt



RESULTS

\- External\_Validation

\- Final\_Hard\_Data\_Split

\- Final\_Model\_Training

\- Final\_Live\_Test\_Evidence



README.md



\---



\## 3. Main Final Program



Main file:



C:\\Do\_Not\_Delete\_PLC\\finalv26\\CODE\\live\_final\_crest\_yolo26.py



This is the finalized live system.



Supporting pallet tracker:



C:\\Do\_Not\_Delete\_PLC\\finalv26\\CODE\\dynamic\_live\_pallet\_detection.py



\---



\## 4. Models Used



\### Pallet Detection Model



File:



pallet\_yolo26n\_best.pt



Purpose:



Detects pallets in each camera frame and provides pallet bounding boxes.



\---



\### EMPTY / OBJECT\_PRESENT Classification Model



File:



empty\_object\_yolo26n\_cls\_best.pt



Purpose:



Determines whether a detected pallet is:



\- EMPTY

\- OBJECT\_PRESENT



\---



\### X / NO\_X Classification Model



File:



x\_no\_x\_yolo26n\_medium\_focus\_final\_hard\_best.pt



Purpose:



For pallets containing an object, this model determines whether the object contains the rejection X mark.



Classification:



\- X -> REJECTED

\- NO\_X -> ACCEPTED



This is the final hard-finetuned YOLO26 symbol classifier.



\---



\## 5. Pallet Tracking



Each detected pallet receives a software track ID.



YOLO provides a bounding box:



(x1, y1, x2, y2)



The pallet center is calculated as:



center\_x = (x1 + x2) / 2



center\_y = (y1 + y2) / 2



The current pallet position is therefore represented in camera image pixels.



For a 1920 x 1080 frame:



\- (0,0) is the top-left corner

\- X increases toward the right

\- Y increases toward the bottom



The tracking system associates pallet detections between frames using information such as:



\- bounding-box overlap

\- center displacement

\- pallet size

\- short-term motion

\- previous track information



The tracker also maintains approximate pallet motion information.



\---



\## 6. Conveyor Boundary



File:



CONFIG\\conveyor\_boundary.json



The conveyor boundary defines the valid conveyor region in the camera image.



Pallet detections outside the valid conveyor area are rejected.



The boundary contains outer and inner conveyor regions so the active conveyor path can be represented as a ring.



\---



\## 7. Final Live Decision Logic



The final system uses temporal information instead of relying on a single camera frame.



\### New Object



When a new object is placed on an EMPTY moving pallet:



1\. Multiple OBJECT\_PRESENT observations confirm that an object was added.

2\. A short delay allows temporary hand obstruction to clear.

3\. X / NO\_X classification begins immediately.

4\. Multiple observations are collected.

5\. The system produces ACCEPTED or REJECTED within a few seconds.



The system does not require the pallet to complete another conveyor lap.



\---



\### Temporary Occlusion



If a pallet already has a reliable result such as:



ACCEPTED



or:



REJECTED



and a hand, wire, or other temporary obstruction appears, the system keeps the previous reliable result.



Temporary obstruction does not automatically reset the pallet to CHECKING.



\---



\### Real Object Removal



If an object is actually removed from a pallet, repeated reliable EMPTY observations are required before the state changes to:



EMPTY



This prevents a short visual obstruction from being mistaken for physical object removal.



\---



\### Object Classification Change



If the physical object changes from X to NO\_X or from NO\_X to X, the system can update the pallet result after receiving sufficient reliable new observations.



\---



\## 8. Symbol Image Processing



The X / NO\_X classifier uses a medium-focus representation of the detected pallet.



The symbol ROI focuses approximately on:



\- horizontal region: 15% to 85% of pallet width

\- vertical region: 0% to 67% of pallet height



The ROI is resized using aspect-ratio-preserving padding before YOLO26 classification.



This representation was selected because it makes the X / NO\_X symbol larger while retaining sufficient surrounding object context.



\---



\## 9. Validation Results



\### Independent External Validation



Image-level result:



58 / 60 correct



Accuracy:



96.67%



Physical-pass majority result:



20 / 20 correct



Accuracy:



100.00%



\---



\### Final Hard Locked Test



Image-level result:



23 / 24 correct



Accuracy:



95.83%



X:



11 / 12 correct



NO\_X:



12 / 12 correct



Precision:



100%



Recall:



91.67%



Specificity:



100%



F1 Score:



95.65%



Physical-pass majority:



8 / 8 correct



Accuracy:



100%



\---



\## 10. Important Final Result Folders



\### External\_Validation



Contains independent external validation evidence collected after the medium-focus model was frozen.



\### Final\_Hard\_Data\_Split



Contains information about the final hard-case train/test split.



\### Final\_Model\_Training



Contains the final YOLO26 hard-finetuning configuration, training results, plots, and evaluation evidence.



\### Final\_Live\_Test\_Evidence



Contains evidence generated while testing the final live conveyor system.



\---



\## 11. Software Environment



Development environment:



C:\\VLM\_Finetune



Virtual environment:



vlm\_env



Main software versions used during final development:



\- Python 3.12.13

\- PyTorch 2.11.0+cu128

\- Ultralytics 8.4.129

\- CUDA 12.8



GPU used during development and training:



NVIDIA GeForce RTX 5090



\---



\## 12. Camera Configuration



Final physical camera configuration used during development:



\- Resolution: 1920 x 1080

\- Frame rate: approximately 30 FPS

\- Exposure: -7.0

\- Auto exposure value: 0.25



The program attempts available camera indexes automatically.



\---



\## 13. How to Run the Final System



Open Command Prompt and run:



cd /d C:\\VLM\_Finetune



vlm\_env\\Scripts\\activate



python "C:\\Do\_Not\_Delete\_PLC\\finalv26\\CODE\\live\_final\_crest\_yolo26.py"



A physical camera must be connected before running the live system.



\---



\## 14. Final Runtime Dependencies



The live system requires these files:



CODE\\live\_final\_crest\_yolo26.py



CODE\\dynamic\_live\_pallet\_detection.py



MODELS\\pallet\_yolo26n\_best.pt



MODELS\\empty\_object\_yolo26n\_cls\_best.pt



MODELS\\x\_no\_x\_yolo26n\_medium\_focus\_final\_hard\_best.pt



CONFIG\\conveyor\_boundary.json



Do not remove or rename these files without also updating the program.



\---



\## 15. Development History



The complete development history, older models, experimental code, failed tests, and intermediate results remain in:



C:\\Do\_Not\_Delete\_PLC\\CREST\_YOLO26



The YOLO26_Final_System folder contains only the files needed for the finalized project and its important supporting evidence.




