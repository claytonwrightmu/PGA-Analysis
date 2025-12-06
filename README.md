**PGA-Analysis**
##
PGA-Analysis is my ongoing build-out of a real golf analytics engine. It’s grown into a multi-season data pipeline, a talent model, and a foundation for full tournament simulations.

The project evolves every time I learn something new, which is exactly the point.

What I’m trying to create
- A system that actually understands golf, not just spreadsheets.
- True talent, not hot streaks
- A multi-year skill model that blends strokes-gained data from different seasons to show who’s consistently elite and who just pops for a few months.

Course DNA

- Every course asks different questions. Distance tolerance, iron demand, green complexity — I want to translate all of that into clean numerical profiles.
- Player–course fit
- Once I know how courses behave and how players score, the next step is matching them.
     Who benefits from firm fairways? Who thrives on approach-heavy setups?
          This project is where I plan to answer that.

Simulation-ready structure

- Everything is being built with the idea that tournament simulations will eventually sit on top of it. Probabilities, cutline projections, future golf ball rollback.
##
**Where the project currently stands**
##
Right now, the pipeline can:

- Merge multi-season data
- Build a master table of all player performance
- Generate a multi-year “true talent” rating
- Produce stable rankings and feature columns for future modeling

The foundation is finally strong enough to keep stacking more complex ideas on top of it.

What’s next
- Upgrade multi-year weighting and feature engineering
- Build the first version of Course DNA
- Score player–course compatibility
- Move toward tournament simulations
