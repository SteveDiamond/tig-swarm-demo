macro_rules! impl_kv_string_serde {
    ($name:ident { $( $field:ident : $ty:ty ),* $(,)? }) => {
        paste::paste! {
            #[derive(Debug, Clone, PartialEq)]
            pub struct $name {
                $( pub $field : $ty ),*
            }

            impl serde::Serialize for $name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    let mut parts = Vec::new();
                    $(
                        parts.push(format!("{}={}", stringify!($field), self.$field));
                    )*
                    // optional: sort keys for deterministic output
                    parts.sort();
                    let s = parts.join(",");
                    serializer.serialize_str(&s)
                }
            }

            impl<'de> serde::Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>
                {
                    use serde::de::{Visitor, Error};
                    use std::fmt;

                    struct VisitorImpl;

                    impl<'de> Visitor<'de> for VisitorImpl {
                        type Value = $name;

                        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            write!(f, "a string of the form 'key=value,key=value'")
                        }

                        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                        where
                            E: Error,
                        {
                            let mut map = std::collections::HashMap::new();

                            if !v.is_empty() {
                                for part in v.split(',') {
                                    let mut kv = part.splitn(2, '=');
                                    let key = kv.next().ok_or_else(|| E::custom(format!("Missing key in '{}'", part)))?;
                                    let val = kv.next().ok_or_else(|| E::custom(format!("Missing value in '{}'", part)))?;
                                    map.insert(key, val);
                                }
                            }

                            Ok($name {
                                $(
                                    $field: map.get(stringify!($field))
                                        .ok_or_else(|| E::custom(format!("Missing field '{}'", stringify!($field))))?
                                        .parse::<$ty>()
                                        .map_err(E::custom)?,
                                )*
                            })
                        }
                    }

                    deserializer.deserialize_str(VisitorImpl)
                }
            }
        }
    };
}

#[cfg(feature = "job_scheduling")]
pub mod job_scheduling;

#[cfg(feature = "knapsack")]
pub mod knapsack;

#[cfg(feature = "vehicle_routing")]
pub mod vehicle_routing;
