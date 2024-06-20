def count_integers_in_file(filename):
    try:
        with open(filename, 'r') as file:
            count = 0
            for line in file:
                try:
                    # Attempt to convert the line to an integer
                    int(line.strip())
                    count += 1
                except ValueError:
                    # If the line cannot be converted to an integer, skip it
                    continue
            return count
    except FileNotFoundError:
        print(f"Failed to open the file '{filename}'.")
        return 0

if __name__ == "__main__":
    filename = "numbers.txt"
    count = count_integers_in_file(filename)
    print(f"The file '{filename}' contains {count} integers.")
