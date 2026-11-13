"""Allowed entity types and directed predicates for the space-operations graph."""

from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    MISSION = "Mission"
    ORGANIZATION = "Organization"
    SPACECRAFT = "Spacecraft"
    LAUNCH_VEHICLE = "LaunchVehicle"
    EQUIPMENT = "Equipment"
    CELESTIAL_BODY = "CelestialBody"
    LOCATION = "Location"
    FACILITY = "Facility"
    EVENT = "Event"
    TEST_CAMPAIGN = "TestCampaign"
    TEST_TECHNIQUE = "TestTechnique"
    ENGINEERING_FINDING = "EngineeringFinding"


class Predicate(str, Enum):
    HAS_EVENT = "HAS_EVENT"
    TESTS = "TESTS"
    HAS_COMPONENT = "HAS_COMPONENT"
    CARRIES = "CARRIES"
    BUILT_BY = "BUILT_BY"
    DEVELOPED_BY = "DEVELOPED_BY"
    PROVIDED_BY = "PROVIDED_BY"
    USES_LAUNCH_VEHICLE = "USES_LAUNCH_VEHICLE"
    LAUNCHED_FROM = "LAUNCHED_FROM"
    LANDS_AT = "LANDS_AT"
    TARGETS = "TARGETS"
    LOCATED_IN = "LOCATED_IN"
    CONDUCTED_AT = "CONDUCTED_AT"
    TESTED_ARTICLE = "TESTED_ARTICLE"
    USES_TECHNIQUE = "USES_TECHNIQUE"
    COMMUNICATES_WITH = "COMMUNICATES_WITH"
    RELAYS_DATA_TO = "RELAYS_DATA_TO"
    HAS_FINDING = "HAS_FINDING"
    ABOUT_VEHICLE = "ABOUT_VEHICLE"


class TemporalStatus(str, Enum):
    PLANNED = "planned"
    EXPECTED = "expected"
    OCCURRED = "occurred"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


_RELATION_ENDPOINTS: dict[
    Predicate, tuple[frozenset[EntityType], frozenset[EntityType]]
] = {
    Predicate.HAS_EVENT: (
        frozenset({EntityType.MISSION}),
        frozenset({EntityType.EVENT}),
    ),
    Predicate.TESTS: (
        frozenset({EntityType.MISSION, EntityType.TEST_CAMPAIGN}),
        frozenset(
            {
                EntityType.SPACECRAFT,
                EntityType.LAUNCH_VEHICLE,
                EntityType.EQUIPMENT,
            }
        ),
    ),
    Predicate.HAS_COMPONENT: (
        frozenset({EntityType.SPACECRAFT, EntityType.LAUNCH_VEHICLE}),
        frozenset({EntityType.SPACECRAFT, EntityType.EQUIPMENT}),
    ),
    Predicate.CARRIES: (
        frozenset({EntityType.SPACECRAFT, EntityType.LAUNCH_VEHICLE}),
        frozenset({EntityType.SPACECRAFT, EntityType.EQUIPMENT}),
    ),
    Predicate.BUILT_BY: (
        frozenset(
            {
                EntityType.SPACECRAFT,
                EntityType.LAUNCH_VEHICLE,
                EntityType.EQUIPMENT,
            }
        ),
        frozenset({EntityType.ORGANIZATION}),
    ),
    Predicate.DEVELOPED_BY: (
        frozenset(
            {
                EntityType.MISSION,
                EntityType.SPACECRAFT,
                EntityType.LAUNCH_VEHICLE,
                EntityType.EQUIPMENT,
            }
        ),
        frozenset({EntityType.ORGANIZATION}),
    ),
    Predicate.PROVIDED_BY: (
        frozenset({EntityType.SPACECRAFT, EntityType.EQUIPMENT}),
        frozenset({EntityType.ORGANIZATION}),
    ),
    Predicate.USES_LAUNCH_VEHICLE: (
        frozenset({EntityType.MISSION, EntityType.EVENT}),
        frozenset({EntityType.LAUNCH_VEHICLE}),
    ),
    Predicate.LAUNCHED_FROM: (
        frozenset({EntityType.EVENT}),
        frozenset({EntityType.FACILITY, EntityType.LOCATION}),
    ),
    Predicate.LANDS_AT: (
        frozenset(
            {EntityType.MISSION, EntityType.EVENT, EntityType.SPACECRAFT}
        ),
        frozenset({EntityType.LOCATION, EntityType.FACILITY}),
    ),
    Predicate.TARGETS: (
        frozenset({EntityType.MISSION}),
        frozenset({EntityType.CELESTIAL_BODY, EntityType.LOCATION}),
    ),
    Predicate.LOCATED_IN: (
        frozenset({EntityType.LOCATION, EntityType.FACILITY}),
        frozenset(
            {EntityType.LOCATION, EntityType.FACILITY, EntityType.CELESTIAL_BODY}
        ),
    ),
    Predicate.CONDUCTED_AT: (
        frozenset({EntityType.TEST_CAMPAIGN}),
        frozenset({EntityType.FACILITY, EntityType.LOCATION}),
    ),
    Predicate.TESTED_ARTICLE: (
        frozenset({EntityType.TEST_CAMPAIGN}),
        frozenset(
            {
                EntityType.SPACECRAFT,
                EntityType.LAUNCH_VEHICLE,
                EntityType.EQUIPMENT,
            }
        ),
    ),
    Predicate.USES_TECHNIQUE: (
        frozenset({EntityType.TEST_CAMPAIGN}),
        frozenset({EntityType.TEST_TECHNIQUE}),
    ),
    Predicate.COMMUNICATES_WITH: (
        frozenset({EntityType.SPACECRAFT}),
        frozenset({EntityType.SPACECRAFT, EntityType.EQUIPMENT}),
    ),
    Predicate.RELAYS_DATA_TO: (
        frozenset({EntityType.SPACECRAFT, EntityType.EQUIPMENT}),
        frozenset({EntityType.SPACECRAFT, EntityType.EQUIPMENT}),
    ),
    Predicate.HAS_FINDING: (
        frozenset({EntityType.TEST_CAMPAIGN, EntityType.EVENT}),
        frozenset({EntityType.ENGINEERING_FINDING}),
    ),
    Predicate.ABOUT_VEHICLE: (
        frozenset({EntityType.ENGINEERING_FINDING}),
        frozenset(
            {
                EntityType.SPACECRAFT,
                EntityType.LAUNCH_VEHICLE,
                EntityType.EQUIPMENT,
            }
        ),
    ),
}


def relation_endpoints_are_valid(
    predicate: Predicate, subject_type: EntityType, object_type: EntityType
) -> bool:
    allowed_subjects, allowed_objects = _RELATION_ENDPOINTS[predicate]
    return subject_type in allowed_subjects and object_type in allowed_objects
