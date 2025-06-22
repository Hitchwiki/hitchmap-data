from pydantic import Field, BaseModel
from typing import List, Optional, Tuple
from enum import Enum


class Location(BaseModel):
    latitude: float = Field(...)
    longitude: float = Field(...)
    is_exact: bool = Field(...)


class MethodEnum(str, Enum):
    thumb = "thumb"
    waving = "waving"
    sign = "sign"
    asking = "asking"
    invited = "invited"
    prearranged = "prearranged"

class Signal(BaseModel):
    methods: List[MethodEnum]
    sign_content: Optional[str] = None
    sign_languages: Optional[List[str]] = None
    asking_content: Optional[str] = None
    asking_languages: Optional[List[str]] = None
    total_solicited: Optional[int] = None
    duration: Optional[str] = None


class ReasonEnum(str, Enum):
    holiday = "holiday"
    commute = "commute"
    business = "business"
    recreational = "recreational"

class Ride(BaseModel):
    vehicle_destination: Optional[Location] = None
    reason: Optional[List[ReasonEnum]] = None


class GenderEnum(str, Enum):
    male = "male"
    female = "female"
    non_binary = "non_binary"
    prefer_not_to_say = "prefer_not_to_say"

class Person(BaseModel):
    origin_location: Optional[str] = None
    origin_country: Optional[str] = None
    year_of_birth: Optional[int] = None
    gender: Optional[GenderEnum] = None
    languages: Optional[List[str]] = None
    was_driver: Optional[bool] = None


class ReasonToPickUpEnum(str, Enum):
    is_hitchhiker = "is_hitchhiker"
    was_hitchhiker = "was_hitchhiker"
    social_exchange = "social_exchange"
    cultural_exchange = "cultural_exchange"
    environmental = "environmental"
    wanted_driver = "wanted_driver"
    curiosity = "curiosity"
    hospitality_norm = "hospitality_norm"
    elevated_mood = "elevated_mood"
    nonthreatening_appearance = "nonthreatening_appearance"
    sympathy = "sympathy"
    safety_concern = "safety_concern"
    opposed = "opposed"

class Occupant(Person):
    reason_to_pick_up: Optional[ReasonToPickUpEnum] = None


class KindEnum(str, Enum):
    car = "car"
    bus = "bus"
    van = "van"
    truck = "truck"
    motorbike = "motorbike"
    scooter = "scooter"
    taxi = "taxi"
    horse_cart = "horse-cart"
    train = "train"
    camper = "camper"
    tractor = "tractor"
    plane = "plane"
    ferry = "ferry"
    boat = "boat"

class ModeOfTranportation(BaseModel):
    kind: KindEnum = Field(...)
    make: Optional[str] = None
    model: Optional[str] = None
    license_plate_country: Optional[str] = None  # ISO 3166-1 alpha-2
    license_plate_identifier: Optional[str] = None


class ReasonsToHitchhikeEnum(str, Enum):
    commute = "commute"
    vacation = "vacation"
    sport = "sport"
    financial = "financial"
    social_exchange = "social_exchange"
    cultural_exchange = "cultural_exchange"
    recreational = "recreational"
    environmental = "environmental"
    fundraising = "fundraising"

class Hitchhiker(Person):
    nickname: Optional[str] = None  # Nickname of the hitchhiker. Assumed unique within the data source.
    hitchhiking_since: Optional[int] = None  # The year the person hitchhiked for the first time.
    reasons_to_hitchhike: Optional[List[ReasonsToHitchhikeEnum]] = None  # Reasons for a specific hitchhiking ride.

class GiftKindEnum(str, Enum):
    money = "money"
    food = "food"
    goods = "goods"

class Gift(BaseModel):
    kind: GiftKindEnum = Field(...)
    description: Optional[str] = None
    price: Optional[Tuple[float, str]] = None  # [amount, currency]

class DeclinedRideReasonEnum(str, Enum):
    wrong_direction = "wrong_direction"
    too_short = "too_short"
    too_long = "too_long"
    risk_concern = "risk_concern"
    safety_concern = "safety_concern"
    space_missing = "space_missing"
    too_slow = "too_slow"

class DeclinedRide(BaseModel):
    destination: Optional[Location] = None
    reasons: Optional[List[DeclinedRideReasonEnum]] = None


class Stop(BaseModel):
    location: Location = Field(...)
    arrival_time: Optional[str] = None  # RFC 9557 format
    departure_time: Optional[str] = None  # RFC 9557 format
    waiting_duration: Optional[str] = None  # ISO 8601 duration format


class HitchhikingRecord(BaseModel):
    stops: List[Stop] = Field(..., min_items=1)
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None
    signals: Optional[List[Signal]] = None
    hitchhikers: List[Hitchhiker] = Field(..., min_items=1)
    occupants: Optional[List[Occupant]] = None
    mode_of_transportation: Optional[ModeOfTranportation] = None
    ride: Optional[Ride] = None
    declined_rides: Optional[List[DeclinedRide]] = None
    source: str = Field(...)
    license: str = Field(...)
    submission_time: Optional[str] = None  # RFC 9557 format