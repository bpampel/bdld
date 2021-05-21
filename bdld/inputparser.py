"""Input class and helpers to parse input from file

The configparser parses only simple items (str, float, int, bool), this was
extended to also allow comma separated list of values. More complicated syntax
like nested lists makes the writing and parsing of input files more error-prone.

The downside is that some of the options might need transforming. This is not done
directly but some helper functions to do it are provided in this file.
"""

import configparser
import logging
from typing import (
    Any,
    cast,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    Tuple,
    Type,
)


BuiltinType = Union[str, float, int, bool]
OptionType = Union[BuiltinType, List[BuiltinType], None]


class OptionError(Exception):
    """Custom error message if option missing or could not be parsed"""

    def __init__(self, message, key, section):
        super().__init__(f"Option '{key}' in section [{section}]: {message}")


class SectionError(Exception):
    """Error if section is missing"""

    def __init__(self, section):
        super().__init__(f"No section '{section}' found, but it is required")


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
        default: OptionType = None,
    ) -> None:
        self.key = key
        self.keytype = keytype
        self.compulsory = compulsory
        self.condition = condition
        self.default = default

    def parse(self, section: configparser.SectionProxy) -> Optional[OptionType]:
        """Parse option from section in exception save way

        Returns either the option as the desired type or None if the key is not found
        """
        val = self.default
        if self.key not in section.keys():
            if self.compulsory:
                e = "Option not found, but it is required"
                raise OptionError(e, self.key, section.name)
            return val
        if self.keytype == str:
            val = section.get(self.key)
        try:
            if self.keytype == float:
                val = section.getfloat(self.key)
            if self.keytype == int:
                val = section.getint(self.key)
            if self.keytype == bool:
                val = section.getbool(self.key)
        except ValueError as e:
            raise OptionError(
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
                raise OptionError(self.condition.desc, self.key, section.name)
        return val


class Input:
    """Class that parses the input file

    All sections are stored into the data dict with the keys being the section names.

    Each section of the config is parsed in a seperate function defining the individual
    InputOption objects. The parsed options are then stored as dict.
    This dict is put as value for the respective section into the nested "data" dict.

    The data dict thus looks like this (2 sections with 2 options each):

        {sec1: {opt1: abc, opt2: edf}, sec2: {opt3: uvw, opt4: xyz}}

    :param filename: config file to parse
    :param data: nested dictionary containing all sections with all parsed options
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
        """Define the data members and call the parsing function"""
        self.filename = filename
        self.data: Dict[str, Dict[str, OptionType]] = {}

        # test that file can be opened -> raises FileNotFoundError if not
        with open(self.filename) as _:
            pass

        print(f"Parsing input from file '{self.filename}'\n")
        self.infile = configparser.ConfigParser()
        self.infile.read(self.filename)

        self.parse_all()

    def parse_all(self) -> None:
        """Read input file and make sure the compulsory sections are included

        Does then launch the config for the individual sections
        """
        required_sections = ["ld", "potential", "particles"]
        optional_sections = [
            "trajectories",
            "histogram",
            "fes",
            "birth-death",
            "delta-f",
            "particle-distribution",
        ]

        # mandatory sections, that can be there only once
        for sec in required_sections:
            if not self.infile.has_section(sec):
                raise SectionError(sec)
            self.parse_section(sec)

        for section_type in optional_sections:
            # multiple of these sections are possible, get all that start with the type
            numbered_secs = [
                sec for sec in self.infile.sections() if sec.find(section_type) == 0
            ]
            for sec in numbered_secs:
                self.parse_section(sec)

        for sec in self.infile.sections():  # only not parsed ones left
            logging.warning(
                "%s",
                f'Warning: Section "{sec}" did not match anything and will be ignored. Is there a typo?',
            )

    def parse_section(self, section_type: str, label: str = None) -> None:
        """Parse a section

        This maps the strings to the respective parsing functions
        containing the actual options.
        The received options are then stored in the "data" dict

        To parse multiple sections of the same type, the "key"
        can be used to parse a specific section of the config

        The corresponding xyz_opts() functions have all the same signature
        and might actually parse the given section to decide which options
        are valid (e.g. based on system dimension)

        :param section_type: base type of section to parse
        :param label: optional section key / label used in config
        """
        if not label:  # default: use just the section type
            label = section_type

        if section_type == "ld":
            options = self.ld_opts(self.infile[label])
        elif section_type == "potential":
            options = self.potential_opts(self.infile[label])
        elif section_type == "particles":
            options = self.particles_opts(self.infile[label])
        elif section_type == "birth-death":
            options = self.birth_death_opts(self.infile[label])
        elif section_type == "trajectories":
            options = self.trajectories_opts(self.infile[label])
        elif section_type == "histogram":
            options = self.histogram_opts(self.infile[label])
        elif section_type == "fes":
            options = self.fes_opts(self.infile[label])
        elif section_type == "delta-f":
            options = self.delta_f_opts(self.infile[label])
        elif section_type == "particle-distribution":
            options = self.particle_distribution_opts(self.infile[label])

        parsed_options: Dict[str, OptionType] = {}
        for o in options:
            parsed_options[o.key] = o.parse(self.infile[label])
            self.infile.remove_option(label, o.key)

        for opt_key in self.infile.options(label):  # all remaining ones
            if opt_key not in self.infile.defaults():
                logging.warning(
                    "%s",
                    f'Warning: Option "{opt_key}" in section "{label} did not match anything and will be ignored. Is there a typo?"',
                )

        self.data[label] = parsed_options
        self.infile.remove_section(label)

    def ld_opts(self, section: configparser.SectionProxy) -> List[InputOption]:
        """Define and parse the options of the langevin dynamics"""
        type_option = InputOption("type", str, False)
        ld_type = cast(str, type_option.parse(section))
        options = [
            type_option,
            InputOption("timestep", float, True, Input.positive_or_zero),
            InputOption("n_steps", int, True, Input.positive),
            InputOption("seed", int, False),
        ]
        if ld_type == "bussi-parinello":
            options += [
                InputOption("kt", float, True, Input.positive),
                InputOption("friction", float, True, Input.positive_or_zero),
            ]
        elif ld_type == "overdamped":
            pass
        else:
            raise OptionError(
                f'Specified ld type "{ld_type}" is not implemented',
                "type",
                section.name,
            )
        return options

    def potential_opts(self, section: configparser.SectionProxy) -> List[InputOption]:
        """Define the options of the potential"""
        type_option = InputOption("type", str, True)
        pot_type = cast(str, type_option.parse(section))
        if pot_type == "polynomial":
            n_dim_option = InputOption("n_dim", int, True, Input.at_most_3dim)
            n_dim = cast(int, n_dim_option.parse(section))
            if n_dim == 1:
                options = [
                    type_option,
                    n_dim_option,
                    InputOption("coeffs", [float], False),
                    InputOption("min", float, True),
                    InputOption("max", float, True),
                ]
            else:
                options = [
                    type_option,
                    n_dim_option,
                    InputOption("coeffs-file", str, True),
                    InputOption("min", [float], True),
                    InputOption("max", [float], True),
                ]
        elif pot_type == "mueller-brown":
            options = [
                type_option,
                InputOption("scaling-factor", float, False),
                InputOption("n_dim", int, False, default=2),
            ]
        else:
            raise OptionError(
                f'Specified potential type "{pot_type}" is not implemented',
                "type",
                section.name,
            )
        options.append(InputOption("boundary-condition", str, False))
        return options

    def particles_opts(self, section: configparser.SectionProxy) -> List[InputOption]:
        """Define options for the particles and initial distribution"""
        options = [
            InputOption("number", int, True, Input.positive),
            InputOption("mass", float, False, Input.positive, 1.0),
            InputOption("seed", int, False),
        ]

        # a bit more logic because the variants require different options
        initial_distribution_variants = [
            "random-global",
            "random-pos",
            "fractions-pos",
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

        if init_dist in ["random-pos", "fractions-pos"]:
            for i, _ in enumerate(get_all_numbered_values(section, "pos")):
                options.append(InputOption(f"pos{i+1}", [float], True))

        if init_dist == "fractions-pos":
            options.append(InputOption("fractions", [float], True))

        return options

    def birth_death_opts(self, section: configparser.SectionProxy) -> List[InputOption]:
        """Define options of the birth-death process"""
        # specify allowed density estimation methods
        eq_dens_methods = [
            None,  # default, will be same as potential
            "potential",
            "uniform",
            "histogram",
        ]
        allowed_eq_dens_methods = Condition(
            lambda x: x in eq_dens_methods,
            f"must be one of {eq_dens_methods}",
        )

        options = [
            InputOption("stride", int, True, Input.positive),
            InputOption("correction-variant", str, False),  # not checked here
            InputOption("equilibrium-density-method", str, False, allowed_eq_dens_methods),
            InputOption("density-estimate-histogram", str, False),
            InputOption("density-estimate-stride", int, False, Input.positive),
            InputOption("stats-stride", int, False, Input.positive),
            InputOption("stats-filename", str, False),
            InputOption("seed", int, False),
        ]

        if self.data["potential"]["n_dim"] == 1:
            options.append(InputOption("kernel-bandwidth", float, True, Input.positive))
        else:
            options.append(
                InputOption("kernel-bandwidth", [float], True, Input.all_positive)
            )
        return options

    def trajectories_opts(
        self, section: configparser.SectionProxy
    ) -> List[InputOption]:
        """Define and parse the options for trajectory output"""
        options = [
            InputOption("filename", str, False),
            InputOption("stride", int, False, Input.positive),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
        ]
        return options

    def histogram_opts(self, section: configparser.SectionProxy) -> List[InputOption]:
        """Define and parse the options for histogramming the trajectories"""
        options = [
            InputOption("stride", int, False, Input.positive),
            InputOption("reset", [int], False, Input.all_positive),
            InputOption("filename", str, False, None),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
        ]
        if self.data["potential"]["n_dim"] == 1:
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
        return options

    def fes_opts(self, section: configparser.SectionProxy) -> List[InputOption]:
        """Define and parse the fes section"""
        if "histogram" not in self.data.keys():
            raise configparser.NoSectionError("histogram")
        options = [
            InputOption("kt", float, True, Input.positive),
            InputOption("stride", int, False, Input.positive),
            InputOption("filename", str, False),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
            InputOption("plot-stride", int, False),
            InputOption("plot-filename", str, False),
            InputOption("plot-domain", [float], False),
            InputOption("plot-title", str, False),
        ]
        return options

    def delta_f_opts(self, section: configparser.SectionProxy) -> List[InputOption]:
        """Define and parse if delta f should be calculated section"""
        if "fes" not in self.data.keys():
            raise configparser.NoSectionError("fes")
        options = [
            InputOption("stride", int, True, Input.positive),
            InputOption("filename", str, False),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
        ] + numbered_state_options(section)
        return options

    def particle_distribution_opts(
        self, section: configparser.SectionProxy
    ) -> List[InputOption]:
        """Define and parse if statistics about particles should be printed periodically"""
        options = [
            InputOption("stride", int, True, Input.positive),
            InputOption("filename", str, False),
            InputOption("write-stride", int, False, Input.positive),
            InputOption("fmt", str, False),
        ] + numbered_state_options(section)
        return options


# helper functions to transform the options
def get_all_numbered_values(
    section: Mapping[str, Any],
    prefix: str = "",
    suffix: str = "",
) -> List[OptionType]:
    """Returns all numbered options/values with the given key name

    :param section: section/dict to search
    :param prefix: prefix of option key (string before number)
    :param suffix: suffix of option key (string after number)
    """
    counter = 1
    res: List[OptionType] = []
    while True:
        try:
            value = section[f"{prefix}{counter}{suffix}"]
        except KeyError:  # not found
            return res
        res.append(value)
        counter += 1


def numbered_state_options(section: configparser.SectionProxy) -> List[InputOption]:
    """Return list with 'state{i}_min' & 'state{i}_max' InputOptions in section

    This reads the keys of a section and adds two new options for each found state
    The state InputOptions are lists regardless of dimensions

    :param section: section of configparser to search through
    """
    n_states = len(get_all_numbered_values(section, "state", "-min"))
    if n_states != len(get_all_numbered_values(section, "state", "-max")):
        raise OptionError(
            "The number of min and max options for the states doesn't match",
            "statex-min",
            section.name,
        )
    options = []
    for i in range(n_states):
        options += [
            InputOption(f"state{i+1}-min", [float], True),
            InputOption(f"state{i+1}-max", [float], True),
        ]
    if not options:
        raise OptionError(
            "No state options found for action",
            "statex-min / statex-max",
            section.name,
        )
    return options


def min_max_to_ranges(
    min_list: Union[List[float], List[List[float]]],
    max_list: Union[List[float], List[List[float]]],
) -> List[List[Tuple[float, float]]]:
    """Transform the max and min lists to a list of lists of tuples for the states

    The min_list and max_list have one point per entry, i.e. for more than 1D they are
    also lists.
    This also checks that all min and max values have the same dimension

    The output list contains one list for every range.
    The list then contains a tuple with (min, max) value for every dimension.

    :param min_list: list with all minimum points of the intervals
    :param max_list: list with all maximum points of the intervals
    """
    n_items = len(min_list)
    if n_items != len(max_list):
        raise ValueError("Not the same number of minimum and maximum points given")

    # check if dimensions of elements match
    try:
        n_dim = len(min_list[0])  # TypeError if not list
        for i, _ in enumerate(min_list):
            if len(min_list[i]) != len(max_list[i]) != n_dim:
                raise ValueError(f"Dimensions of min/max {i} are inconsistent")
    except TypeError:  # elements are not lists
        for min_max_i in min_list + max_list:
            float(min_max_i)  # verify that none is list
        # put all items in lists
        min_list = [[item] for item in min_list]
        max_list = [[item] for item in max_list]

    return [list(zip(min_list[i], max_list[i])) for i in range(n_items)]
