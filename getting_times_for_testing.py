import time

# Get current time
current_time = time.time()

# Subtract 3 hours (3 hours = 3 * 60 * 60 seconds)
hours_ago = current_time - (3 * 60 * 60)

print("Current Time:", current_time)
print("Time Hours Ago:", hours_ago)
