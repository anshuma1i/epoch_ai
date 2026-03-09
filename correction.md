## Incorrect interpretation of observer_position

Dear participants,

We have received a question regarding the feature observer_position in the training set. The participant questioned why most of the time the position of the observer was done at dozens of meters above the sea level.

Turns out that this label was wrongly indicated as the location of the observer encoded in EWKB (Extended Well-Known Binary) Hex as Longitude / Latitude / Altitude. This feature actually corresponds to the most recent bird position in the radar track when the observer clicked the button to enter the species label to the track, encoded in EWKB.

We apologize for the confusion, the data description is now updated with this change.