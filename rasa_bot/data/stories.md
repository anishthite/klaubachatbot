<!-- ## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye -->

## say goodbye
* goodbye
  - utter_goodbye

## bot challenge
* bot_challenge
  - utter_iamabot

## ask directions
* greet_give_directions
  - utter_clarify
* affirm
  - utter_happy


## ask directions
* greet
  - utter_greet_ask_directions
* give_direction
  - slot{"room":"value"}
  - utter_clarify
> user_affirm
* affirm
  - utter_happy
> user_affirm
* deny
    - utter_clarify
> user_affirm