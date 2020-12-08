"""Config class and helpers to parse input from file"""

import configparser
from typing import (
    cast,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)


BuiltinType = Union[str, float, int, bool]
OptionType = Union[BuiltinType, List[BuiltinType]]


class ConfigError(Exception):
    """Custom error message style related to config parsing"""

    def __init__(self, message, key, section):
        super().__init__(f"Option '{key}' in section [{section}]: {message}")


class Condition:
    """Combine a string description with the evaluating function of a condition"""

    def __init__(self, function: Callable, desc: str):
        self.function = function
        self.desc = desc


class ConfigOption:
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
                raise ConfigError(
                    "Could not be found but is required", self.key, section.name
                )
            return None
        if self.keytype == str:
            res: OptionType = section.get(self.key)
        try:
            if self.keytype == float:
                res = section.getfloat(self.key)
            if self.keytype == int:
                res = section.getint(self.key)
            if self.keytype == bool:
                res = section.getbool(self.key)
        except ValueError as e:
            raise ConfigError(
                f"could not be converted to {self.keytype.__name__}",
                self.key,
                section.name,
            ) from e
        # parsing of lists is a bit more complicated
        if isinstance(self.keytype, list):
            res = section.get(self.key)
            res = [self.keytype[0](val) for val in res.split(",")]
        if self.condition:
            if not self.condition.function(res):
                raise ConfigError(self.condition.desc, self.key, section.name)
        return res


class Config:
    """Class that parses the input file and contains dictionaries for all the config options"""

    positive = Condition(lambda x: x > 0, "must be greater than zero")
    positive_or_zero = Condition(lambda x: x >= 0, "must be greater or equal than zero")
    at_most_3dim = Condition(lambda x: x > 0 & x < 4, "must be between 1 and 3")

    def __init__(self, infile: str) -> None:
        self.infile = infile
        # each config sections has it's own dictionary
        self.ld: Dict[str, OptionType] = {}
        self.potential: Dict[str, OptionType] = {}
        self.particles: Dict[str, OptionType] = {}
        self.birth_death: Optional[Dict[str, OptionType]] = None
        self.histogram: Optional[Dict[str, OptionType]] = None

        self.parse_all()

    def parse_all(self) -> None:
        """Read input file and make sure the compulsory sections are included

        Does then launch the config for the individual sections
        """
        config = configparser.ConfigParser()
        config.read(self.infile)

        for required_sec in ["ld", "potential", "particles"]:
            if not required_sec in config.sections():
                e = f"No {required_sec} section in input, but it is required"
                raise configparser.NoSectionError(e)

        self.parse_ld(config["ld"])
        self.parse_potential(config["potential"])
        self.parse_particles(config["particles"])

    def parse_ld(self, section: configparser.SectionProxy) -> None:
        """Define and parse the options of the langevin dynamics"""
        options = [
            ConfigOption("timestep", float, True, Config.positive_or_zero),
            ConfigOption("n_steps", int, True, Config.positive),
            ConfigOption("temperature", float, True, Config.positive),
            ConfigOption("friction", float, True, Config.positive_or_zero),
        ]
        self.ld = self.parse_section(section, options)

    def parse_potential(self, section: configparser.SectionProxy) -> None:
        """Define and parse the options of the potential"""
        n_dim_option = ConfigOption("n_dim", int, True, Config.at_most_3dim)
        n_dim = cast(int, n_dim_option.parse(section))

        if n_dim == 1:
            options = [
                n_dim_option,
                ConfigOption("coeffs", [float], True, None),
                ConfigOption("min", float, True, None),
                ConfigOption("max", float, True, None),
            ]
        else:
            options = [
                n_dim_option,
                ConfigOption("min", [float], True, None),
                ConfigOption("max", [float], True, None),
            ]
            for i in range(n_dim):
                options.append(ConfigOption("coeffs" + str(i + 1), [float], True, None))
        self.potential = self.parse_section(section, options)

    def parse_particles(self, section: configparser.SectionProxy) -> None:
        """Parse the number of particles and initial distribution"""
        initial_distribution_variants = [
            "random-global",
            "random-states",
            "equal-states",
        ]
        allowed_initial_distribution = Condition(
            lambda x: x in initial_distribution_variants,
            f"must be one of {initial_distribution_variants}",
        )

        options = [
            ConfigOption("number", int, True, Config.positive),
            ConfigOption(
                "initial-distribution", str, True, allowed_initial_distribution
            ),
        ]
        self.particles = self.parse_section(section, options)

    @staticmethod
    def parse_section(
        section: configparser.SectionProxy, options: List[ConfigOption]
    ) -> Dict[str, OptionType]:
        """Parse all options of a section

        Logic prevents all non-compulsory options from being added to the dictionary"""
        parsed_options: Dict[str, OptionType] = {}
        for o in options:
            res = o.parse(section)
            if res is not None:
                parsed_options[o.key] = res
        return parsed_options
