import setuptools

setuptools.setup(
        name="astro-nba",
        version="0.1",
        author="Nicolas Garavito Camargo",
        author_email="ngaravito@flatironinstitute.org",
        description="Analysis of idealized N-body simulations using Python",
        packages=["nba", "nba/ios", "nba/com", "nba/structure",\
                "nba/kinematics", "nba/orbits", "nba/visuals"]
        )
