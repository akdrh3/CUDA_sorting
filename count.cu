int main()
{
    std::ifstream inputFile("numbers.txt");
    if (!inputFile)
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    int number;
    int count = 0;
    while (inputFile >> number)
    {
        count++;
    }
    inputFile.close();

    std::cout << "The file contains " << count << " integers." << std::endl;

    return 0;
}