import rebound
import numpy as np


def read_planet_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    planets = []
    for i in range(1, n + 1):
        data = list(map(float, lines[i].strip().split()))
        mass = data[0]
        pos = data[1:4]
        vel = data[4:7]
        planets.append((mass, pos, vel))

    integration_time = float(lines[n + 1].strip())
    time_interval = float(lines[n + 2].strip())

    return planets, integration_time, time_interval


def simulate(planets, integration_time, time_interval, output_file):
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")

    for mass, pos, vel in planets:
        sim.add(m=mass, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])

    sim.move_to_com()

    times = np.arange(0, integration_time + time_interval, time_interval)
    results = []

    for t in times:
        sim.integrate(t)
        snapshot = []
        for particle in sim.particles:
            snapshot.append(
                [
                    particle.x,
                    particle.y,
                    particle.z,
                    particle.vx,
                    particle.vy,
                    particle.vz,
                ]
            )
        results.append((t, snapshot))

    with open(output_file, "w") as f:
        for t, snapshot in results:
            f.write(f"Time: {t}\n")
            for planet_data in snapshot:
                f.write(" ".join(map(str, planet_data)) + "\n")
            f.write("\n")


if __name__ == "__main__":
    input_file = "planets_data.txt"
    output_file = "integration_results.txt"

    planets, integration_time, time_interval = read_planet_data(input_file)
    simulate(planets, integration_time, time_interval, output_file)
