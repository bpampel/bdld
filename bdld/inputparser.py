"""Input class and helpers to parse input from file

The configparser parses only simple items (str, float, int, bool), this was
extended to also allow comma separated list of values. More complicated syntax
like nested lists makes the writing and parsing of input files more error-prone.

The downside is that some of the options might need transforming. This is not done
directly but some helper functions to do it are provided in this file.
"""

import configparser
from typing import (
    cast,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Type,
)


BuiltinType = Union[str, float, int, bool]
OptionType = Union[BuiltinType, List[BuiltinType], None]


class InputError(Exception):
    """Custom error message style related to config parsing"""

    def __init__(self, message, key, section):
        super().__init__(f"Option '{key}' in section [{section}]: {message}")


class Condition:
    """Combine a string description with the evaluating function of a condition"""

    def __init__(self, function: Callable, desc: str):
        self.function = function
        self.desc = desc


class InputOption:
    """Bundle information about a config option into class

    :param key: keyword in config
    :param keytype: expected type (one of OptionType)"""

    def __init__(
        self,
        key: str,
        keytype: Union[Type[BuiltinType], List[Type[BuiltinType]]],
        compulsory: bool = False,
        condition: Optional[Condition] = None,
    ) -> None:
        self.key = key
        self.keytype = keytype
        self.compulsory = compulsory
        self.condition = condition

    def parse(self, section: configparser.SectionProxy) -> Optional[OptionType]:
        """Parse option from section in exception save way

        Returns either the option as the desired type or None if the key is not found
        """
        if self.key not in section.keys():
            if self.compulsory:
                raise InputError(
                    "Could not be found but is required", self.key, section.name
                )
            return None
        if self.keytype == str:
            val: OptionType = section.get(self.key)
        try:
            if self.keytype == float:
                val = section.getfloat(self.key)
            if self.keytype == int:
                val = section.getint(self.key)
            if self.keytype == bool:
                val = section.getbool(self.key)
        except ValueError as e:
            raise InputError(
                f"could not be converted to {self.keytype.__name__}",
                self.key,
                section.name,
            ) from e
        # parsing of lists is a bit more complicated
        if isinstance(self.keytype, list):
            val = section.get(self.key)
            val = [self.keytype[0](x) for x in val.split(",")]
        if self.condition:
            if not self.condition.function(val):
                raise InputError(self.condition.desc, self.key, section.name)
        return val


class Input:
    """Class that parses the input file

    Each section of the config is parsed in a seperate function defining the individual
    InputOption objects. The parsed options are then stored in one dictionary per
    section.

    :paam
    """

    # define conditions here
    positive = Condition(lambda x: x > 0, "must be greater than zero")
    all_positive = Condition(
        lambda lst: all(x > 0 for x in lst), "all values must be greater than zero"
    )
    positive_or_zero = Condition(lambda x: x >= 0, "must be greater or equal than zero")
    at_most_3dim = Condition(lambda x: x > 0 & x < 4, "must be between 1 and 3")

    # @staticmethod
    # def match_dimensions(n_dim: int) -> Condition:
    # return Condition(lambda x: x == n_dim, f"wrong dimensions (must be {n_dim})")

    def __init__(self, filename: str) -> None:
        self.filename = filename
        # each config sections has it's own dictionary
        self.ld: Dict[str, OptionType] = {}
        self.potential: Dict[str, OptionType] = {}
        self.particles: Dict[str, OptionType] = {}
        self.birth_death: Optional[Dict[str, OptionType]] = None
        self.trajectories: Optional[Dict[str, OptionType]] = None
        self.histogram: Optional[Dict[str, OptionType]] = None
        self.fes: Optional[Dict[str, OptionType]] = None
        self.delta_f: Optional[Dict[str, OptionType]] = None
        self.particle_statistics: Optional[Dict[str, OptionType]] = None

        self.parse_all()

    def parse_all(self) -> None:
        """Read input file and make sure the compulsory sections are included

        Does then launch the config for the individual sections
        """
        # test that file can be opened -> raises FileNotFoundError if not
        with open(self.filename) as _:
            pass

        print(f"Parsing input from file '{self.filename}'\n")
        infile = configparser.ConfigParser()
        infile.read(self.filename)

        for required_sec in ["ld", "potential", "particles"]:
            if not required_sec in infile.sections():
                e = f"No {required_sec} section in input, but it is required"
                raise configparser.NoSectionError(e)

        self.parse_ld(infile["ld"])
        self.parse_potential(infile["potential"])
        self.parse_particles(infile["particles"])

        if infile.has_section("birth-death"):
            self.parse_birth_death(infile["birth-death"])
        if infile.has_section("trajectories"):
            self.parse_trajectories(infile["trajectories"])
        if infile.has_section("histogram"):
            self.parse_histogram(infile["histogram"])
        if infile.has_section("fes"):
            self.parse_fes(infile["fes"])
        if infile.has_section("delta-f"):
            self.parse_delta_f(infile["delta-f"])
        if infile.has_section("particle-statistics"):
            self.parse_particle_statistics(infile["particle-statistics"])

    def parse_ld(self, section: configparser.SectionProxy) -> None:
        """Define and parse the options of the langevin dynamics"""
        options = [
            InputOption("timestep", float, True, Input.positive_or_zero),
            InputOption("n_steps", int, True, Input.positive),
            InputOption("kt", float, True, Input.positive),
            InputOption("friction", float, True, Input.positive_or_zero),
            InputOption("seed", int, False),
        ]
        self.ld = self.parse_section(section, options)

    def parse_potential(self, section: configparser.SectionProxy) -> None:
        """Define and parse the options of the potential"""
        n_dim_option = InputOption("n_dim", int, True, Input.at_most_3dim)
        n_dim = cast(int, n_dim_option.parse(section))

        if n_dim == 1:
            options = [
                n_dim_option,
                InputOption("coeffs", [float], True),
                InputOption("min", float, True),
                InputOption("max", float, True),
            ]
        else:
            options = [
                n_dim_option,
                InputOption("min", [float], True),
                InputOption("max", [float], True),
            ]
            for i in range(n_dim):
                options.append(InputOption("coeffs" + str(i + 1), [float], True))
        options.append(InputOption("seed", int, False))
        self.potential = self.parse_section(section, options)

    def parse_particles(self, section: configparser.SectionProxy) -> None:
        """Parse the number of particles and initial distribution"""
        options = [
            InputOption("number", int, True, Input.positive),
            InputOption("seed", int, False),
        ]

        # a bit more logic because the variants require different options
        initial_distribution_variants = [
            "random-global",
            "random-pos",
            "fraction-pos",
        ]
        allowed_initial_distribution = Condition(
            lambda x: x in initial_distribution_variants,
            f"must be one of {initial_distribution_variants}",
        )

        init_dist_option = InputOption(
            "initial-distribution", str, True, allowed_initial_distribution
        )
        init_dist = cast(str, init_dist_option.parse(section))
        options.append(init_dist_option)

        if init_dist in ["random-pos", "fraction-pos"]:
            for i in range(self.get_number_of_options(section, "pos")):
                options.append(InputOption(f"pos{i}", [float], True))

        if init_dist == "fractions-pos":
            options.append(InputOption("fractions", [float], True))

        self.particles = self.parse_section(section, options)

    def parse_birth_death(self, section: configparser.SectionProxy) -> None:
        """Define and parse the options of the potential"""
        options = [
            InputOption("stride", int, True, Input.positive),
            InputOption("correction-variant", str, False),  # not checked here
            InputOption("stats-stride", int, False, Input.positive),
            InputOption("seed", int, False),
        ]

        if self.potential["n_dim"] == 1:
            options.append(InputOption("kernel-bandwidth", float, True, Input.positive))
        else:
            options.append(
                InputOption("kernel-bandwidth", [float], True, Input.all_positive)
            )
        self.birth_death = self.parse_section(section, options)

    def parse_trajectories(self, section: configparser.SectionProxy) -> None:
        """Define and parse the options for trajectory output"""
        options = [
            InputOption("filename", str, False),
            InputOption("stride", int, False, Input.positive),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
        ]
        self.trajectories = self.parse_section(section, options)

    def parse_histogram(self, section: configparser.SectionProxy) -> None:
        """Define and parse the options for histogramming the trajectories"""
        options = [
            InputOption("stride", int, False, Input.positive),
            InputOption("filename", str, False, None),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
        ]
        if self.potential["n_dim"] == 1:
            options += [
                InputOption("min", float, True),
                InputOption("max", float, True),
                InputOption("bins", int, True, Input.positive),
            ]
        else:
            options += [
                InputOption("min", [float], True),
                InputOption("max", [float], True),
                InputOption("bins", [int], True, Input.all_positive),
            ]
        self.histogram = self.parse_section(section, options)

    def parse_fes(self, section: configparser.SectionProxy) -> None:
        """Define and parse the fes section"""
        if not self.histogram:
            raise configparser.NoSectionError("histogram")
        options = [
            InputOption("kt", float, True, Input.positive),
            InputOption("stride", int, False, Input.positive),
            InputOption("filename", str, False),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
            InputOption("plot-stride", int, False),
            InputOption("plot-filename", str, False),
            InputOption("plot-domain", str, False),
            InputOption("plot-title", str, False),
        ]
        self.fes = self.parse_section(section, options)

    def parse_delta_f(self, section: configparser.SectionProxy) -> None:
        """Define and parse if delta f should be calculated section"""
        if not self.fes:
            raise configparser.NoSectionError("fes")
        options = [
            InputOption("stride", int, True, Input.positive),
        ]
        n_states = len(get_all_numbered_values(section, "state", "-min"))
        if n_states == len(get_number_of_options(section, "state", "-max")):
            raise InputError(
                "The number of min and max options for the states doesn't match",
                "statex-min",
                section.name,
            )
        for i in range(n_states):
            options += [
                InputOption("state" + str(i + 1) + "-min", float, True),
                InputOption("state" + str(i + 1) + "-max", float, True),
            ]
        self.delta_f = self.parse_section(section, options)

    def parse_particle_statistics(self, section: configparser.SectionProxy) -> None:
        """Define and parse if statistics about particles should be printed periodically"""
        options = [
            InputOption("stride", int, True, Input.positive),
        ]
        n_states = len(get_all_numbered_values(section, "state", "-min"))
        if n_states == len(get_all_numbered_values(section, "state", "-max")):
            raise InputError(
                "The number of min and max options for the states doesn't match",
                "statex-min",
                section.name,
            )
        for i in range(n_states):
            options += [
                InputOption("state" + str(i + 1) + "-min", float, True),
                InputOption("state" + str(i + 1) + "-max", float, True),
            ]
        self.particle_statistics = self.parse_section(section, options)

    @staticmethod
    def parse_section(
        section: configparser.SectionProxy, options: List[InputOption]
    ) -> Dict[str, OptionType]:
        """Parse all options of a section"""
        parsed_options: Dict[str, OptionType] = {}
        for o in options:
            parsed_options[o.key] = o.parse(section)
        return parsed_options


# helper functions to transform the options
def get_all_numbered_values(
    section: Union[configparser.SectionProxy, Dict[str, OptionType]],
    prefix: str = "",
    suffix: str = "",
) -> List[OptionType]:
    """Returns all numbered options with the given key name

    :param section: section/dict to search
    :param prefix: prefix of option key (string before number)
    :param suffix: suffix of option key (string after number)
    """
    counter = 0
    res: List[OptionType] = []
    while True:
        value = section[f"{prefix}{counter+1}{suffix}"]
        if value:
            res.append(value)
            counter += 1
        else:
            return res


def min_max_to_ranges(
    min_list: List[float], max_list: List[float]
) -> List[Tuple[float, float]]:
    """Transform the max and min lists to a single list of tuples

    The min_list and max_list have one point per entry, i.e. for more than 1D they are
    also lists.

    :param min_list: list with all minimum points of the intervals
    :param max_list: list with all maximum points of the intervals
    """
    if len(min_list) != len(max_list):
        raise ValueError("Not the same number of minimum and maximum points given")
    return list(zip(min_list, max_list))
