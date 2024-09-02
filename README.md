# OrcaChessEngine

OrcaChessEngine is an advanced AI-powered chess engine that combines cutting-edge machine learning techniques with traditional chess algorithms to provide a powerful and intelligent chess-playing experience.

![Orca Chess Engine Logo]([https://i.natgeofe.com/n/87a36612-27e8-4e6b-b188-82c37a8dd95a/NationalGeographic_2772395.jpg])
## Features

- **Strong AI Opponent**: Play against a formidable AI that adapts to your skill level.
- **Opening Book**: Extensive library of chess openings for diverse gameplay.
- **Endgame Tablebases**: Perfect play in various endgame scenarios.
- **Multi-threading Support**: Utilizes full CPU power for faster analysis.
- **UCI Compatible**: Can be used with popular chess GUIs.
- **Position Analysis**: Get detailed evaluations of any chess position.
- **Game Analysis**: Review your games with AI-powered insights.
- **Customizable Playing Strength**: Adjust the AI's strength to match your level.

## Installation

### Prerequisites

- C++ Compiler (GCC 7.0+ or equivalent)
- CMake 3.10+
- CUDA Toolkit 10.0+ (for GPU acceleration)

### Building from Source

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/OrcaChessEngine.git
   cd OrcaChessEngine
   ```

2. Create a build directory:
   ```
   mkdir build && cd build
   ```

3. Configure and build:
   ```
   cmake ..
   make
   ```

4. The executable `orca_chess` will be created in the `build` directory.

## Usage

### Command Line

Run OrcaChessEngine from the command line:

```
./orca_chess
```

Use UCI commands to interact with the engine.

### With a Chess GUI

1. Open your preferred chess GUI (e.g., Arena, Cutechess).
2. Add OrcaChessEngine as a new engine.
3. Start a new game with OrcaChessEngine as your opponent.

## Configuration

Edit the `config.ini` file to customize engine settings:

- `Threads`: Number of CPU threads to use
- `Hash`: Size of hash table in MB
- `UCI_LimitStrength`: Enable/disable strength limiting
- `UCI_Elo`: Set approximate playing strength

## Contributing

We welcome contributions to OrcaChessEngine! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Deep Blue team for pioneering chess AI
- Stockfish project for open-source chess engine insights
- AlphaZero paper for revolutionary AI techniques in chess

## Contact

For support or queries, please open an issue on this repository or contact us at support@orcachess.com.

---

Happy chess playing with OrcaChessEngine!
