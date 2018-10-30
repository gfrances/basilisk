

def add_bw_domain_parameters(language):
    # We simply add block "a" as a domain constant
    return [language.constant("a", "object")]
    # language.constant("b", "object")


def add_bw_domain_parameters_2(language):
    return [language.constant("a", "object"), language.constant("b", "object")]


def no_parameter(lang):
    return []


def update_dict(d, **kwargs):
    res = d.copy()
    res.update(kwargs)
    return res
