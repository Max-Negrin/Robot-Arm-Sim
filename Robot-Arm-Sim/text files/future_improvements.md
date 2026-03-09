


Add functionality to offset each joint so that the planes along which the two links rotate are evenly spaced out, and the end effector always reaches the target.


add documentation for the json file's structure in the documentation file

For the json file - add syntax that records the arm's information (ie the number and length of the links, as well as their offsets from each other, and other information required (starting angle, to name another)) that can be recorded. It should still be able to hold waypoints too.


Add the collision margin to the gui somehow- the arm should never intersect itself, and it should never touch below the xy plane, as that is the table. add a vertical offset functionality for the baseto the joint plane offsets tab menu


In higher values of N, the links >joint 3 are all locked in a straight line, not bending. the other joints should be moving, even though this is mostly a cosmetic change.

adding the link/joint offsets is essential for user experience, and it is also extremely important that with this change, the IK is still accurate.


implement the joint plane offsets into the GUI

Update the handoff and documentation file accordingly

