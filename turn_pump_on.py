import testmodel

def turn_pump_on(output):
    
    #print
    if str(output).strip() == "0 wilted": #if wilted, turn pump on
        return True
    elif str(output).strip() == "1 watered": #if watered, do not turn pump on
        return False
    else:
        print("Unexpected output ->", repr(output)) #sanity check to prevent crashing
# turn_pump_on() # Run the function

# Output
if __name__ == "__main__":
    output = testmodel.WOW()
    if turn_pump_on(output):
        print("Turn Pump on")
    else:
        print("Do not turn pump on")


